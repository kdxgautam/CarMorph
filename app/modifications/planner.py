"""Select the least-powerful renderer capable of the requested edit."""

from app.errors import PipelineError
from app.modifications.schemas import RendererMode, SurfaceEditRequest
from app.renderers.base import ModificationRenderer
from app.renderers.deterministic import DeterministicSurfaceRenderer
from app.renderers.generative import GenerativeSurfaceRenderer


def choose_renderer(modification: SurfaceEditRequest) -> ModificationRenderer:
    """Choose deterministic rendering unless requested features require FLUX."""

    if modification.renderer == RendererMode.DETERMINISTIC:
        if modification.design_elements or modification.custom_instruction:
            raise PipelineError(
                "renderer_not_supported",
                "Deterministic renderer supports body colour and finish only",
            )
        return DeterministicSurfaceRenderer()
    if modification.renderer == RendererMode.GENERATIVE:
        return GenerativeSurfaceRenderer()
    if modification.design_elements or modification.custom_instruction:
        return GenerativeSurfaceRenderer()
    return DeterministicSurfaceRenderer()
