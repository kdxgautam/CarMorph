import re
from typing import Protocol

from app.errors import PipelineError
from app.modifications.schemas import (
    PaintFinish,
    RacingStripeElement,
    StripeAlignment,
    StripePlacement,
    StripeWidth,
    SurfaceEditRequest,
    normalise_hex,
)

COLOURS = {
    "black": "#111111",
    "white": "#ffffff",
    "red": "#d61f2c",
    "blue": "#183a63",
    "grey": "#777777",
    "gray": "#777777",
    "silver": "#c0c0c0",
    "yellow": "#f6c445",
    "green": "#147a3d",
}
PHYSICAL_WORDS = re.compile(
    r"\b(rims?|wheels?|tyres?|tires?|bumper|spoiler|suspension|body kit|"
    r"convertible|lower|bigger tyres|bigger tires)\b",
    re.I,
)


class InstructionParser(Protocol):
    def parse(self, instruction: str) -> SurfaceEditRequest:
        ...


def _colour(text: str, after: str = "") -> str | None:
    segment = text[text.find(after) + len(after) :] if after and after in text else text
    match = re.search(r"#?[0-9a-fA-F]{6}", segment)
    if match:
        return normalise_hex(match.group(0))
    for name, value in COLOURS.items():
        if re.search(rf"\b{name}\b", segment):
            return value
    return None


class RestrictedInstructionParser:
    def parse(self, instruction: str) -> SurfaceEditRequest:
        text = instruction.strip().lower()
        if not text:
            raise PipelineError("unsupported_instruction", "Instruction is empty")
        if PHYSICAL_WORDS.search(text):
            raise PipelineError(
                "future_physical_modification",
                "Physical modifications are not supported in this milestone",
            )

        finish = next(
            (item for item in PaintFinish if re.search(rf"\b{item.value}\b", text)),
            PaintFinish.GLOSSY,
        )
        body_text = text.split("with", 1)[0] if "stripe" in text and "with" in text else text
        body_colour = _colour(body_text)
        stripes = []
        if "stripe" in text:
            count = 2 if re.search(r"\b(two|dual|double|2)\b", text) else 1
            width = next(
                (item for item in StripeWidth if re.search(rf"\b{item.value}\b", text)),
                StripeWidth.THIN,
            )
            placement = (
                StripePlacement.BONNET_AND_VISIBLE_ROOF
                if "roof" in text and "bonnet" in text
                else StripePlacement.VISIBLE_ROOF
                if "roof" in text
                else StripePlacement.VISIBLE_SIDE_PANELS
                if "side" in text
                else StripePlacement.BONNET
            )
            alignment = (
                StripeAlignment.LOWER_SIDE if "lower side" in text else StripeAlignment.CENTRE
            )
            stripes.append(
                RacingStripeElement(
                    count=count,
                    colour=_colour(text, "stripe") or "#ffffff",
                    width=width,
                    placement=placement,
                    alignment=alignment,
                )
            )

        if not body_colour and not stripes:
            raise PipelineError(
                "unsupported_instruction",
                "Instruction did not match supported surface paint changes",
            )
        return SurfaceEditRequest(
            body_colour=body_colour,
            finish=finish,
            design_elements=stripes,
            custom_instruction=instruction.strip(),
        )


def merge_instruction(modification: SurfaceEditRequest) -> SurfaceEditRequest:
    if not modification.custom_instruction:
        return modification
    parsed = RestrictedInstructionParser().parse(modification.custom_instruction)
    return modification.model_copy(
        update={
            "body_colour": modification.body_colour or parsed.body_colour,
            "finish": parsed.finish
            if modification.finish == PaintFinish.GLOSSY and parsed.finish != PaintFinish.GLOSSY
            else modification.finish,
            "design_elements": modification.design_elements or parsed.design_elements,
        }
    )
