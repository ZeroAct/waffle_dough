import logging
import os

import dotenv
from waffle_utils.file import io

dotenv.load_dotenv()


class Config:
    # Dataset Configuration
    DATASET_ROOT_DIR: str = os.environ.get("WAFFLE_DATASET_ROOT_DIR", "./datasets")


logger = logging.getLogger(__name__)

io.make_directory(Config.DATASET_ROOT_DIR)
logger.info(f"Dataset root directory: {Config.DATASET_ROOT_DIR}")
