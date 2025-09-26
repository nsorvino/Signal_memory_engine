# utils/logging_setup.py
import logging
import os
import sys


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = '%(asctime)s level=%(levelname)s name=%(name)s msg="%(message)s"'
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
