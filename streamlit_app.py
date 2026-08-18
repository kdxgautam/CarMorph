"""Interactive car paint and constrained bumper-preview editor."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from app.bumper_analysis.reference_preprocessor import store_bumper_reference
from app.config import GenerativeSettings, Settings
from app.errors import PipelineError
from app.generative.vertex_ai import vertex_provider
from app.modifications.schemas import BumperReplacementRequest, RimReplacementRequest, StudioRenderRequest, SurfaceEditRequest
from app.pipeline import process_view
from app.renderers.deterministic import DeterministicSurfaceRenderer
from app.renderers.generative_bumper import GenerativeBumperRenderer
from app.renderers.generative_rim import GenerativeRimRenderer
from app.renderers.generative_studio import GenerativeStudioRenderer
from app.rim_analysis import store_rim_reference

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="Car Paint Studio", page_icon=":material/directions_car:", layout="wide")
st.title("Car Paint Studio")
st.caption("Natural, mask-protected car paint and bumper-preview renders.")

for key, value in {
    "processed_source_hash": None,
    "processed_view_selection": None,
    "processed_metadata": None,
    "processed_storage_root": None,
    "composition_history": [],
    "pending_result_bytes": None,
    "pending_result_metadata": None,
    "pending_result_kind": None,
}.items():
    st.session_state.setdefault(key, value)


def clear_processed() -> None:
    for key in (
        "processed_source_hash", "processed_view_selection", "processed_metadata",
        "processed_storage_root",
    ):
        st.session_state[key] = None
    clear_composition()


def clear_composition() -> None:
    st.session_state.composition_history = []
    st.session_state.pending_result_bytes = None
    st.session_state.pending_result_metadata = None
    st.session_state.pending_result_kind = None


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def init_composition(directory: Path, metadata) -> None:
    if st.session_state.composition_history:
        return
    with Image.open(directory / metadata.original_image) as image:
        st.session_state.composition_history = [{
            "image_bytes": png_bytes(image),
            "kind": "Original",
            "quality_status": "passed",
            "warnings": [],
            "cached": False,
        }]


def working_bytes() -> bytes | None:
    history = st.session_state.composition_history
    return history[-1]["image_bytes"] if history else None


def working_base_image() -> Image.Image | None:
    history = st.session_state.composition_history
    if len(history) <= 1:
        return None
    return Image.open(BytesIO(history[-1]["image_bytes"])).convert("RGB")


def set_pending(kind: str, result) -> None:
    st.session_state.pending_result_bytes = Path(result.path).read_bytes()
    st.session_state.pending_result_metadata = result
    st.session_state.pending_result_kind = kind


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
                clear_composition()
                init_composition(Path(settings.storage_root) / metadata.asset_id, metadata)
            except PipelineError as exc:
                st.error(f"{exc.detail} ({exc.code})")
            except (OSError, ValueError) as exc:
                st.error(f"Could not process this image: {exc}")

    metadata = st.session_state.processed_metadata
    if source is not None:
        st.image(source, caption="Original", width="content")
    if metadata:
        directory = Path(st.session_state.processed_storage_root) / metadata.asset_id
        init_composition(directory, metadata)
        if metadata.requested_view == "auto":
            st.caption(f"Detected view: {metadata.view} ({metadata.view_confidence:.0%} confidence)")
        history = st.session_state.composition_history
        if history:
            st.caption("Kept changes: " + " → ".join(item["kind"] for item in history))
            undo_column, reset_column, download_column = st.columns(3)
            if undo_column.button(
                "Undo last change",
                icon=":material/undo:",
                disabled=len(history) <= 1,
            ):
                st.session_state.composition_history = history[:-1]
                st.session_state.pending_result_bytes = None
                st.session_state.pending_result_metadata = None
                st.session_state.pending_result_kind = None
                st.rerun()
            if reset_column.button(
                "Reset to original",
                icon=":material/restart_alt:",
                disabled=len(history) <= 1 and st.session_state.pending_result_bytes is None,
            ):
                st.session_state.composition_history = history[:1]
                st.session_state.pending_result_bytes = None
                st.session_state.pending_result_metadata = None
                st.session_state.pending_result_kind = None
                st.rerun()
            download_column.download_button(
                "Download current PNG",
                working_bytes(),
                file_name=f"{Path(uploaded.name).stem if uploaded else metadata.asset_id[:8]}-customised.png",
                mime="image/png",
                icon=":material/download:",
                disabled=working_bytes() is None,
            )
        options = ["Paint", "Studio Render"]
        if metadata.available_modifications.bumper_replacement:
            options.append("Bumper replacement")
        if metadata.available_modifications.rim_replacement:
            options.append("Rim replacement")
        if len(options) > 1:
            mode = st.segmented_control("Modification", options, default="Paint")
        else:
            mode = "Paint"
            st.caption("Bumper and rim replacement need the matching detected masks for this MVP.")

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
                        base_image=working_base_image(),
                    )
                    set_pending("Paint", result)
                except PipelineError as exc:
                    st.error(f"{exc.detail} ({exc.code})")
        elif mode == "Bumper replacement":
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
                                base_image=working_base_image(),
                            )
                            status.update(label="Bumper preview complete", state="complete")
                        set_pending("Bumper preview", result)
                    except PipelineError as exc:
                        st.error(f"{exc.detail} ({exc.code})")
        elif mode == "Studio Render":
            with st.form("studio-controls"):
                studio_style = st.selectbox("Studio style", ("light_studio", "dark_studio", "premium_gradient"))
                preserve_plate = st.checkbox("Preserve number plate", value=True)
                studio_submitted = st.form_submit_button("Generate studio render", icon=":material/photo_camera:", type="primary")
            if studio_submitted:
                try:
                    with st.status("Generating studio render", expanded=True) as status:
                        result = GenerativeStudioRenderer(vertex_provider(GenerativeSettings.from_env())).render(
                            directory=directory,
                            metadata=metadata,
                            modification=StudioRenderRequest(style=studio_style, preserve_plate=preserve_plate),
                            base_image=working_base_image(),
                        )
                        status.update(label="Studio render complete", state="complete")
                    set_pending("Studio Render", result)
                except PipelineError as exc:
                    st.error(f"{exc.detail} ({exc.code})")
        else:
            with st.container(border=True):
                st.subheader("Rim replacement")
                rim_upload = st.file_uploader("Reference rim", type=("jpg", "jpeg", "png", "webp"), key="rim_reference")
                if rim_upload is not None:
                    st.image(rim_upload.getvalue(), caption="Reference rim to use", width="content")
                st.caption("Use a clean single-rim image, preferably transparent PNG or plain background.")
                st.info("This is a visual preview and does not guarantee physical compatibility.", icon=":material/info:")
                with st.form("rim-controls"):
                    rim_submitted = st.form_submit_button(
                        "Generate rim preview", icon=":material/auto_awesome:", type="primary", disabled=rim_upload is None
                    )
                if rim_submitted and rim_upload is not None:
                    try:
                        with st.status("Preparing rim preview", expanded=True) as status:
                            reference = store_rim_reference(directory=directory, metadata=metadata, source=rim_upload.getvalue())
                            status.update(label="Generating rim preview with Vertex AI")
                            result = GenerativeRimRenderer(vertex_provider(GenerativeSettings.from_env())).render(
                                directory=directory,
                                metadata=metadata,
                                modification=RimReplacementRequest(reference_asset_id=reference.reference_asset_id),
                                base_image=working_base_image(),
                            )
                            status.update(label="Rim preview complete", state="complete")
                        set_pending("Rim preview", result)
                    except PipelineError as exc:
                        st.error(f"{exc.detail} ({exc.code})")

        pending = st.session_state.pending_result_bytes
        result = st.session_state.pending_result_metadata
        current = working_bytes()
        if pending and result and current:
            current_column, result_column = st.columns(2)
            current_column.image(current, caption="Current composition", width="stretch")
            result_column.image(pending, caption=st.session_state.pending_result_kind, width="stretch")
            keep_column, discard_column = result_column.columns(2)
            if keep_column.button("Keep this change", icon=":material/check:", type="primary"):
                st.session_state.composition_history.append({
                    "image_bytes": pending,
                    "kind": st.session_state.pending_result_kind,
                    "quality_status": result.quality_status,
                    "warnings": result.warnings,
                    "cached": result.cached,
                })
                st.session_state.pending_result_bytes = None
                st.session_state.pending_result_metadata = None
                st.session_state.pending_result_kind = None
                st.rerun()
            if discard_column.button("Discard preview", icon=":material/close:"):
                st.session_state.pending_result_bytes = None
                st.session_state.pending_result_metadata = None
                st.session_state.pending_result_kind = None
                st.rerun()
            result_column.download_button(
                "Download preview PNG", pending,
                file_name=f"{Path(uploaded.name).stem if uploaded else metadata.asset_id[:8]}-{st.session_state.pending_result_kind.lower().replace(' ', '-')}.png",
                mime="image/png", icon=":material/download:",
            )
            st.success(f"Quality check: {result.quality_status.replace('_', ' ')}")
            if result.cached:
                st.caption("Loaded from cache")
            for warning in (*metadata.warnings, *result.warnings):
                st.warning(warning)
        elif current and len(st.session_state.composition_history) > 1:
            original_column, current_column = st.columns(2)
            original_column.image(st.session_state.composition_history[0]["image_bytes"], caption="Original", width="stretch")
            current_column.image(current, caption="Current composition", width="stretch")

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
