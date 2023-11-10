from .base_type import BaseType


class SplitType(BaseType):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BACKGROUND = "background"
    UNLABELED = "unlabeled"
    PREDICTION = "prediction"
    UNSET = "unset"
