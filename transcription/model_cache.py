from transcription.model_config import ModelSize
from faster_whisper import WhisperModel
import logging


class ModelCache:
    _downloaded_models = {}
    @classmethod
    def add_downloaded_model(cls, model_size: ModelSize, model: WhisperModel):
        cls._downloaded_models[model_size] = model
        logging.info(f"{model_size} added to cache")

    @classmethod
    def check_model_download(cls, model_size: ModelSize):
        return model_size in cls._downloaded_models.keys()

    @classmethod
    def get_model(cls, model_size: ModelSize):
        try:
            model = cls._downloaded_models[model_size]
            logging.info(f"{model_size} retrieved from cache")
            return model
        except KeyError:
            return None