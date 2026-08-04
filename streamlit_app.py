import json
from pathlib import Path

import streamlit as st

from app.config import Settings
from app.errors import PipelineError
from app.modifications.schemas import SurfaceEditRequest
from app.pipeline import process_view
from app.renderers.deterministic import DeterministicSurfaceRenderer

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="Car Paint Studio", page_icon="🚗", layout="wide")
st.title("Car Paint Studio")
st.caption("Natural, mask-protected car paint previews.")

single_tab, evaluation_tab = st.tabs(("Customise one image", "Evaluation gallery"))
with single_tab:
    uploaded = st.file_uploader("Car photo", type=("jpg", "jpeg", "png", "webp"))
    with st.form("paint-controls"):
        view = st.selectbox("Visible car view", ("right", "left", "front", "rear"))
        colour = st.color_picker("Body colour", "#183A63")
        finish = st.selectbox("Paint finish", ("glossy", "matte", "metallic"))
        submitted = st.form_submit_button(
            "Recolour car", type="primary", disabled=uploaded is None
        )

    original_column, result_column = st.columns(2)
    if uploaded is not None:
        original_column.image(uploaded, caption="Original", width="stretch")

    if submitted and uploaded is not None:
        source = uploaded.getvalue()
        if len(source) > MAX_UPLOAD_BYTES:
            st.error("Image exceeds the 20 MB limit.")
        else:
            try:
                with st.spinner("Detecting paint and protected car parts…"):
                    settings = Settings.from_env()
                    metadata = process_view(source, settings, view)
                    result = DeterministicSurfaceRenderer().render(
                        directory=settings.storage_root / metadata.asset_id,
                        metadata=metadata,
                        modification=SurfaceEditRequest(
                            body_colour=colour,
                            finish=finish,
                            renderer="deterministic",
                        ),
                    )
                    rendered = Path(result.path).read_bytes()
            except PipelineError as exc:
                st.error(f"{exc.detail} ({exc.code})")
            except (OSError, ValueError) as exc:
                st.error(f"Could not process this image: {exc}")
            else:
                result_column.image(
                    rendered, caption="Recoloured", width="stretch"
                )
                result_column.download_button(
                    "Download PNG",
                    rendered,
                    file_name=f"{Path(uploaded.name).stem}-recoloured.png",
                    mime="image/png",
                )
                st.success(f"Quality check: {result.quality_status.replace('_', ' ')}")
                for warning in (*metadata.warnings, *result.warnings):
                    st.warning(warning)

with evaluation_tab:
    report_path = Path("evaluation/report.json")
    report = []
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in report}
    inputs = sorted(Path("evaluation/input").glob("*.png"))
    passed = sum(item["status"] == "passed" for item in report)
    st.metric("Pipeline results", f"{passed}/{len(inputs)} passed")
    for source_path in inputs:
        item = by_name.get(source_path.name, {"status": "pending"})
        st.subheader(source_path.name)
        original_column, result_column = st.columns(2)
        original_column.image(
            source_path,
            caption=f"Original · {item.get('view', 'not tested')} view",
            width="stretch",
        )
        if item["status"] == "passed":
            result_column.image(
                Path("evaluation/output") / Path(item["output"]).name,
                caption="Recoloured · #183A63 glossy",
                width="stretch",
            )
            result_column.caption(
                f"Quality: {item['quality'].replace('_', ' ')} · "
                f"{len(item['warnings'])} diagnostic warning(s)"
            )
        elif item["status"] == "failed":
            result_column.error(f"{item['error']} ({item['code']})")
        else:
            result_column.info("Pending pipeline test")
