from enum import StrEnum


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class Modality(StrEnum):
    SENTINEL_2_L2A = "l2a"
    SENTINEL_1_ASC = "asc"
    SENTINEL_1_DES = "des"


class DatasetInstance(StrEnum):
    REGIONAL = "regional"
    RANDOM = "random"
