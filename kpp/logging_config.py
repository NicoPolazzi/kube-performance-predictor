import logging
import sys


def setup_logging(name: str, log_file: str | None = None) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
