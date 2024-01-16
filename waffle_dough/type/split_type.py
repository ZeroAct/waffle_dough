from waffle_utils.enum import StrEnum as BaseType


class SplitType(BaseType):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BACKGROUND = "background"
    UNLABELED = "unlabeled"
    PREDICTION = "prediction"
    UNSET = "unset"
