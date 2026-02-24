from utils import BaseTrainingPipeline

from src.models.coop import CoOP


class CoOPTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "CoOP"
    DEFAULT_OUTPUT_DIR = "outputs/coop"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/coop"
    TRAINER_CLASS = CoOP
