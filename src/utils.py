"""
Utility functions and project configurations for Anomaly NIDS.
"""

import os
import random
import logging
import numpy as np

# Global Configuration Constants
RANDOM_SEED = 42
SAMPLE_FRAC = 0.2  # 20% stratified sample; set to 1.0 for full dataset

# Directory Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

def ensure_dirs():
    """Ensure all required project directories exist."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        NOTEBOOKS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def setup_logger(name: str = "NIDS", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a consistent console and file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
