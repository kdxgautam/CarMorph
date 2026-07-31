from app.modifications.schemas import (
    RacingStripeElement,
    SurfaceEditRequest,
)


def build_surface_prompt(modification: SurfaceEditRequest) -> str:
    finish = modification.finish.value
    colour = modification.body_colour or "the existing base colour"
    parts = [
        "Repaint only the main body paint group.",
        f"Use {finish} automotive paint for the body colour {colour}.",
        "Preserve contrasting roof paint, black pillars, contrasting mirror caps, "
        "contrasting handles, chrome, black plastic, silver trim and glass unless "
        "the structured request explicitly targets them.",
        "Preserve the exact car identity, body geometry, wheels, tyres, lights, "
        "windows, grille, badges, number plate, panel gaps, background, camera "
        "angle, perspective, lighting direction, and shadows.",
        "Do not add, remove, resize, or replace physical parts.",
    ]
    if modification.roof_colour:
        parts.append(
            "Repaint only the contrast roof paint group in "
            f"{modification.roof_colour}, independently from the body."
        )
    for element in modification.design_elements:
        if isinstance(element, RacingStripeElement):
            parts.append(
                f"Add exactly {element.count} {element.width.value} racing stripe"
                f"{'s' if element.count > 1 else ''} in {element.colour} on "
                f"{element.placement.value}, aligned {element.alignment.value}. "
                "Keep stripes perspective-aware and only on editable painted "
                "panels; no paint may appear on protected regions."
            )
    return " ".join(parts)
