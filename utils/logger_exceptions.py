
import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formats every log line as a single JSON object.
    Required for CloudWatch Logs Insights queries.
    Example output:
      {"timestamp": "2024-01-01T10:00:00Z", "level": "INFO",
       "message": "Chat request", "module": "chat",
       "request_id": "abc-123", "query_length": 42}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "request_id": getattr(record, "request_id", None),
        }
        # Attach any extra fields passed via extra={} in logger calls
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                log_obj[key] = value

        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a structured JSON logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("User message received", extra={"request_id": "abc"})
        logger.error("Chain failed", extra={"error": str(e)})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False   # Prevent duplicate logs

    logger.setLevel(logging.INFO)
    return logger