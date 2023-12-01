import numpy as np
import torch
from transcription.model_config import ModelSize
from transcription.model_cache import ModelCache

import whisper

class WhisperSpeechTranscriber:
    def __init__(self, model_size: ModelSize, language_code=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = language_code
        self.model = self.initialize_model(model_size,device)
        self.counter = 1

    @staticmethod
    def initialize_model(model_size: ModelSize, device):
        model_size=model_size.value
        model_in_cache = ModelCache.check_model_download(model_size=model_size)
        if not model_in_cache:
            model = whisper.load_model(model_size, device=device)
            ModelCache.add_downloaded_model(model_size=model_size, model=model)
            return model
        else:
            model = ModelCache.get_model(model_size=model_size)
        return model

    def get_transcription(self, audio: np.ndarray):


        segments = self.model.transcribe(
            audio
        )
        self.counter += 1
        return segments
