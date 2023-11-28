import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
from pyannote.audio import Audio
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Model
from pyannote.audio import Inference
from scipy.spatial.distance import cdist


from utils.audio_utils import np_to_wav

class PyannoteDiarizer():
    """
    Pipeline for speaker diarization
    TODO: The feature extraction is already done in the diarization pipeline, so we should use that instead of
    extracting the features again
    """
    def __init__(self, device="cpu", feature_clustering_threshold = 0.9, ema_alpha = 0.8, from_previous_session=None):
        """
        :param device: device to run the model on
        :param feature_clustering_threshold: threshold for clustering the features
        :param ema_alpha: alpha for exponential moving average for the feature update
        """
        self.segmentation_model = Model.from_pretrained("pyannote/segmentation-3.0", 
                              use_auth_token="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")
        
        self.embedding_model = Model.from_pretrained("pyannote/embedding", 
                                    use_auth_token="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")
        
        self.segmentation_model.to(device)

        self.embedding_model.to(device)

        self.vad = VoiceActivityDetection(segmentation=self.segmentation_model)

        self.vad.instantiate({'min_duration_on': 0.5, 'min_duration_off': 0.0})   
        
        self.embedder = Inference(self.embedding_model, window="whole")
        
        self.num_speakers = 0

        self.ema_alpha = ema_alpha

        self.audio = Audio(sample_rate=16000, mono="downmix")

        self.features = {}

        self.feature_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        self.feature_clustering_threshold = feature_clustering_threshold

    def run(self, chunk: np.ndarray):
        """
        :param chunk: audio chunk
        :return: Annotation object with speaker labels and segments

        Diarize the chunk and feature match the speakers with the previous chunks
        Determine if the speaker was already seen or not in the previous chunks
        """

        segments = self.get_segments(chunk)
        embeddings = self.get_embeddings(chunk, segments)
        speaker_labels = self.cluster_embeddings(embeddings)

        assert len(speaker_labels) == len(segments)

        return speaker_labels, segments
    
    def cluster_embeddings(self, embeddings: np.ndarray):
        """
        :param embeddings: embeddings of the current chunk
        :return: speaker labels for the current chunk

        Does incremental clustering of the embeddings, comparing them to the previous embeddings and
        assigning them to the closest cluster, or creating a new one if the distance is too large
        The comparison is done using cosine distance
        """
        
        speaker_labels = []
        for i, embedding in enumerate(embeddings):
            min_distance = np.inf
            for key, previous_embedding in self.features.items():
                distance = cdist(embedding, previous_embedding, metric="cosine")[0, 0]
                if distance < min_distance:
                    min_distance = distance
                    candidate = key
            if min_distance < self.feature_clustering_threshold:
                speaker_labels.append(candidate)
                if distance < 0.7:
                    self.features[key] = [self.ema_alpha * previous_embedding[0] + (1 - self.ema_alpha) * embedding[0]]
            else:
                self.features[self.feature_labels[self.num_speakers]] = embedding
                speaker_labels.append(self.feature_labels[self.num_speakers])
                self.num_speakers += 1

        return speaker_labels
        
    def get_segments(self, chunk: np.ndarray):
        """
        :param chunk: audio chunk
        :return: diarization object
        """
        segments = self.vad({'waveform': chunk, 'sample_rate': 16000})
        return segments
    
    def segment_embedding(self, chunk: np.ndarray, segment: Segment):
        return [self.embedder({'waveform': chunk[:, int(segment.start * 16000):int(segment.end * 16000)], 'sample_rate': 16000})]
    
    def get_embeddings(self, chunk: np.ndarray, segments):
        embeddings = []
        for track in segments.itertracks(yield_label=True):
            segment = track[0]
            embeddings.append(self.segment_embedding(chunk, segment))
        return embeddings