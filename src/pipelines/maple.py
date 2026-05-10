from utils import BaseTrainingPipeline

from src.models.maple import MaPLe


class MaPLeTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "MaPLe"
    SAVE_BEST_LAST = False
    DEFAULT_OUTPUT_DIR = "outputs/maple"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/maple"
    TRAINER_CLASS = MaPLe
