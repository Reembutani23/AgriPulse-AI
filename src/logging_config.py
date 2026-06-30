import logging
import os

def get_logger(name: str = "agri_pulse") -> logging.Logger:
    """Create and configure a logger.

    Reads LOG_LEVEL from environment (default INFO) and logs to both console and
    a rotating file under `logs/agri_pulse.log`.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level, logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (rotating)
    from logging.handlers import RotatingFileHandler
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/agri_pulse.log", maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("Logger initialized with level %s", log_level)
    return logger
