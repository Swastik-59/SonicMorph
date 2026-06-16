import logging
import sys


def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(lvl)
    if not root.handlers:
        root.addHandler(handler)


if __name__ == "__main__":
    setup_logging("DEBUG")
    logging.getLogger(__name__).info("Logging configured")
