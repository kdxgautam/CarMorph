class PipelineError(Exception):
    def __init__(self, code: str, detail: str, status_code: int = 422) -> None:
        self.code = code
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)
