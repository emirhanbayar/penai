import numpy as np
import torch.cuda

from transcription.model_config import ModelSize, DEVICE, COMPUTATION_TYPE
from transcription.model_cache import ModelCache
from utils.transcription_utils import format_transcription

from faster_whisper import WhisperModel
import stable_whisper


class FasterWhisperSpeechTranscriber:
    def __init__(
        self,
        model_size: ModelSize,
        device="cpu",
        language_code=None,
        compute_type="float32",
        beam_size=1,
    ):
        if(torch.cuda.is_available()):
            device = "cuda"
            compute_type=COMPUTATION_TYPE
        self.language = language_code
        self.model = self.initialize_model(model_size, device, compute_type)
        self._buffer = ""
        self.current_transcription = None
        self.beam_size = beam_size
        self.counter = 1

    @staticmethod
    def initialize_model(model_size: ModelSize, device: str, compute_type: str):
        model_in_cache = ModelCache.check_model_download(model_size=model_size)
        if not model_in_cache:
            model = WhisperModel(
                model_size.value, device=device, compute_type=compute_type
            )
            ModelCache.add_downloaded_model(model_size=model_size, model=model)
            return model
        else:
            model = ModelCache.get_model(model_size=model_size)
        return model

    def inference(self, audio: np.ndarray, **kwargs):
        self.current_transcription = self.get_transcription(audio)
        return self.current_transcription

    def get_transcription(self, audio: np.ndarray):
        """Transcribe audio using Whisper"""

        segments, info = self.model.transcribe(
            audio,
            initial_prompt=self._buffer,
            word_timestamps=True,
            beam_size=self.beam_size,
        )

        segments = list(segments)

        transcription = format_transcription(segments, info)
        self.counter += 1
        return transcription

    def transcribe(self, audio: np.ndarray):

        try:
            aligned_transcription = stable_whisper.transcribe_any(
                inference_func=self.inference, audio=audio, input_sr=16000
            ).to_dict()
            if aligned_transcription['text'] == "":

                return self.current_transcription
        except Exception as e:

            return self.current_transcription

        return aligned_transcription

    def real_time_transcription(self, audio: np.ndarray):
        audio=audio.numpy()
        transcription = self.transcribe(audio)
        # Update transcription buffer
        self._buffer += transcription["text"]
        return transcription
