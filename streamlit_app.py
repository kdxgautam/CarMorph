"""Interactive car paint and constrained bumper-preview editor."""

import hashlib
import json
from pathlib import Path

import streamlit as st

from app.bumper_analysis.reference_preprocessor import store_bumper_reference
from app.config import GenerativeSettings, Settings
from app.errors import PipelineError
from app.generative.vertex_ai import vertex_provider
from app.modifications.schemas import BumperReplacementRequest, SurfaceEditRequest
from app.pipeline import process_view
from app.renderers.deterministic import DeterministicSurfaceRenderer
from app.renderers.generative_bumper import GenerativeBumperRenderer

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="Car Paint Studio", page_icon=":material/directions_car:", layout="wide")
st.title("Car Paint Studio")
st.caption("Natural, mask-protected car paint and bumper-preview renders.")

for key, value in {
    "processed_source_hash": None,
    "processed_view_selection": None,
    "processed_metadata": None,
    "processed_storage_root": None,
    "current_result_bytes": None,
    "current_result_metadata": None,
    "current_result_kind": None,
}.items():
    st.session_state.setdefault(key, value)


def clear_processed() -> None:
    for key in (
        "processed_source_hash", "processed_view_selection", "processed_metadata",
        "processed_storage_root", "current_result_bytes", "current_result_metadata",
        "current_result_kind",
    ):
        st.session_state[key] = None


single_tab, evaluation_tab = st.tabs(("Customise one image", "Evaluation gallery"))
with single_tab:
    uploaded = st.file_uploader("Car photo", type=("jpg", "jpeg", "png", "webp"), key="car_photo")
    with st.form("process-car", border=False):
        view = st.selectbox("Visible car view", ("auto", "front", "rear", "left", "right"))
        process_submitted = st.form_submit_button(
            "Process car", icon=":material/car_repair:", type="primary", disabled=uploaded is None
        )

    source = uploaded.getvalue() if uploaded is not None else None
    source_hash = hashlib.sha256(source).hexdigest() if source else None
    if source_hash and st.session_state.processed_source_hash and (
        source_hash != st.session_state.processed_source_hash
        or view != st.session_state.processed_view_selection
    ):
        clear_processed()

    if process_submitted and source is not None:
        if len(source) > MAX_UPLOAD_BYTES:
            st.error("Image exceeds the 20 MB limit.")
        else:
            try:
                with st.status("Processing car image", expanded=True) as status:
                    settings = Settings.from_env()
                    metadata = process_view(source, settings, view)
                    status.update(label="Car image processed", state="complete")
                st.session_state.processed_source_hash = source_hash
                st.session_state.processed_view_selection = view
                st.session_state.processed_metadata = metadata
                st.session_state.processed_storage_root = settings.storage_root
                st.session_state.current_result_bytes = None
            except PipelineError as exc:
                st.error(f"{exc.detail} ({exc.code})")
            except (OSError, ValueError) as exc:
                st.error(f"Could not process this image: {exc}")

    metadata = st.session_state.processed_metadata
    if source is not None:
        st.image(source, caption="Original", width="content")
    if metadata:
        if metadata.requested_view == "auto":
            st.caption(f"Detected view: {metadata.view} ({metadata.view_confidence:.0%} confidence)")
        if metadata.available_modifications.bumper_replacement:
            mode = st.segmented_control("Modification", ["Paint", "Bumper replacement"], default="Paint")
        else:
            mode = "Paint"
            st.caption("Bumper replacement is available only for front and rear views with a detected bumper in this MVP.")

        directory = Path(st.session_state.processed_storage_root) / metadata.asset_id
        if mode == "Paint":
            with st.form("paint-controls"):
                colour = st.color_picker("Body colour", "#183A63")
                finish = st.selectbox("Paint finish", ("glossy", "matte", "metallic"))
                paint_submitted = st.form_submit_button("Recolour car", icon=":material/format_paint:", type="primary")
            if paint_submitted:
                try:
                    result = DeterministicSurfaceRenderer().render(
                        directory=directory,
                        metadata=metadata,
                        modification=SurfaceEditRequest(body_colour=colour, finish=finish),
                    )
                    st.session_state.current_result_bytes = Path(result.path).read_bytes()
                    st.session_state.current_result_metadata = result
                    st.session_state.current_result_kind = "recoloured"
                except PipelineError as exc:
                    st.error(f"{exc.detail} ({exc.code})")
        else:
            with st.container(border=True):
                st.subheader(f"{metadata.view.capitalize()} bumper replacement")
                reference_upload = st.file_uploader(
                    "Reference bumper", type=("jpg", "jpeg", "png", "webp"), key="bumper_reference"
                )
                if reference_upload is not None:
                    st.image(reference_upload.getvalue(), caption="Reference bumper to use", width="content")
                st.caption("Use a clean front-facing or rear-facing bumper image, preferably a transparent PNG.")
                st.info("This is a visual preview and does not guarantee physical compatibility.", icon=":material/info:")
                with st.form("bumper-controls"):
                    paint_mode = st.segmented_control(
                        "Bumper paint", ["Match body", "Preserve reference"], default="Match body"
                    )
                    bumper_submitted = st.form_submit_button(
                        "Generate bumper preview", icon=":material/auto_awesome:", type="primary", disabled=reference_upload is None
                    )
                if bumper_submitted and reference_upload is not None:
                    reference_bytes = reference_upload.getvalue()
                    try:
                        with st.status("Preparing bumper preview", expanded=True) as status:
                            reference = store_bumper_reference(
                                directory=directory,
                                metadata=metadata,
                                source=reference_bytes,
                                settings=None,
                            )
                            status.update(label="Generating bumper preview with Vertex AI")
                            provider = vertex_provider(GenerativeSettings.from_env())
                            result = GenerativeBumperRenderer(provider).render(
                                directory=directory,
                                metadata=metadata,
                                modification=BumperReplacementRequest(
                                    bumper_position=metadata.view,
                                    reference_asset_id=reference.reference_asset_id,
                                    paint_mode=("match_body" if paint_mode == "Match body" else "preserve_reference"),
                                ),
                            )
                            status.update(label="Bumper preview complete", state="complete")
                        st.session_state.current_result_bytes = Path(result.path).read_bytes()
                        st.session_state.current_result_metadata = result
                        st.session_state.current_result_kind = "bumper-preview"
                    except PipelineError as exc:
                        st.error(f"{exc.detail} ({exc.code})")

        rendered = st.session_state.current_result_bytes
        result = st.session_state.current_result_metadata
        if rendered and result:
            original_column, result_column = st.columns(2)
            original = source or directory / metadata.original_image
            original_column.image(original, caption="Original", width="stretch")
            result_column.image(rendered, caption=st.session_state.current_result_kind.replace("-", " ").capitalize(), width="stretch")
            result_column.download_button(
                "Download PNG", rendered,
                file_name=f"{Path(uploaded.name).stem}-{st.session_state.current_result_kind}.png",
                mime="image/png", icon=":material/download:",
            )
            st.success(f"Quality check: {result.quality_status.replace('_', ' ')}")
            if result.cached:
                st.caption("Loaded from cache")
            for warning in (*metadata.warnings, *result.warnings):
                st.warning(warning)

with evaluation_tab:
    report_path = Path("evaluation/report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else []
    by_name = {item["name"]: item for item in report}
    inputs = sorted(Path("evaluation/input").glob("*.png"))
    passed = sum(item["status"] == "passed" for item in report)
    st.metric("Pipeline results", f"{passed}/{len(inputs)} passed")
    for source_path in inputs:
        item = by_name.get(source_path.name, {"status": "pending"})
        st.subheader(source_path.name)
        original_column, result_column = st.columns(2)
        original_column.image(source_path, caption=f"Original · {item.get('view', 'not tested')} view", width="stretch")
        if item["status"] == "passed":
            result_column.image(Path("evaluation/output") / Path(item["output"]).name, caption="Recoloured · #183A63 glossy", width="stretch")
        elif item["status"] == "failed":
            result_column.error(f"{item['error']} ({item['code']})")
        else:
            result_column.info("Pending pipeline test")
