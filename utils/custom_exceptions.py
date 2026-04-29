import traceback


class CustomException(Exception):
    """
    Base exception class for the project.
    Captures original exception + traceback.
    """

    def __init__(self, message: str, original: Exception = None):
        super().__init__(message)
        self.original = original
        self.trace = traceback.format_exc() if original else None

    def __str__(self):
        base_message = super().__str__()

        if self.original:
            return (
                f"{base_message}\n"
                f"Caused by: {type(self.original).__name__} - {self.original}"
            )

        return base_message

    def full_trace(self):
        """Return full error message + traceback (for logging)"""
        return f"{self.__str__()}\n{self.trace or ''}"


# ─────────────────────────────────────────────
# Specific Exceptions
# ─────────────────────────────────────────────

class DataCleaningException(CustomException):
    """Raised during data cleaning errors"""
    pass


class DataQualityException(CustomException):
    """Raised when data quality checks fail"""
    pass


class S3UploadException(CustomException):
    """Raised when S3 upload fails"""
    pass