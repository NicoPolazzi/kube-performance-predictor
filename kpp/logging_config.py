import logging
import sys


def setup_logging(
    name: str, log_file: str | None = None, level: int = logging.INFO
) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    # Attach handlers to the root logger so every logger in the process can reach them.
    root = logging.getLogger()
    if not root.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    root.setLevel(level)

    return logging.getLogger(name)
