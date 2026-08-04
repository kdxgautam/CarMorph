"""Stable domain errors shared by the API, UI, and processing pipeline."""


class PipelineError(Exception):
    """Expected pipeline failure with a stable client code and HTTP status."""

    def __init__(self, code: str, detail: str, status_code: int = 422) -> None:
        """Store public error fields while retaining normal exception behavior."""

        self.code = code
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)
