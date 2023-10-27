from .base_type import BaseType


class SplitType(BaseType):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNLABELED = "unlabeled"
    PREDICTION = "prediction"
    UNSET = "unset"
