import logging
import os
import sys

_RESET = "\033[0m"
_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.relativepath = os.path.relpath(record.pathname)
        message = super().format(record)
        color = _COLORS.get(record.levelno, "")
        if color and self._supports_color():
            return f"{color}{message}{_RESET}"
        return message

    @staticmethod
    def _supports_color() -> bool:
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


_FMT = "%(asctime)s | %(levelname)-8s | %(relativepath)s:%(lineno)d | %(message)s"


def get_logger(name: str = "ddpm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_ColorFormatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


logger = get_logger()

if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
