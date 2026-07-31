from app.modifications.schemas import PaintFinish, RacingStripeElement, SurfaceEditRequest


def build_surface_prompt(modification: SurfaceEditRequest) -> str:
    finish = modification.finish.value
    colour = modification.body_colour or "the existing base colour"
    parts = [
        "Edit only validated exterior painted surface regions of this exact car.",
        f"Use {finish} automotive paint for the body colour {colour}.",
        "Preserve the exact car identity, body geometry, wheels, tyres, lights, windows, grille, badges, number plate, chrome and black trim, panel gaps, background, camera angle, perspective, lighting direction, and shadows.",
        "Do not add, remove, resize, or replace physical parts.",
    ]
    for element in modification.design_elements:
        if isinstance(element, RacingStripeElement):
            parts.append(
                f"Add exactly {element.count} {element.width.value} racing stripe"
                f"{'s' if element.count > 1 else ''} in {element.colour} on "
                f"{element.placement.value}, aligned {element.alignment.value}. "
                "Keep stripes perspective-aware and only on editable painted panels; no paint may appear on protected regions."
            )
    return " ".join(parts)
