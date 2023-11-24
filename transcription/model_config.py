from enum import Enum

DEVICE = 'cuda'

COMPUTATION_TYPE = "float16"

AUDIO_TYPE = "float32"


class ModelSize(Enum):
    TINY = 'tiny'
    TINY_ENGLISH = 'tiny.en'
    BASE = 'base'
    BASE_ENGLISH = 'base.en'
    SMALL = 'small'
    SMALL_ENGLISH = 'small.en'
    MEDIUM = 'medium'
    MEDIUM_ENGLISH = 'medium.en'
    LARGE_V1 = 'large-v1'
    LARGE_V2 = 'large-v2'


MODELS_FOR_NON_ENGLISH = [ModelSize.LARGE_V1, ModelSize.LARGE_V2]