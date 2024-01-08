from typing import Any
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from pyannote.audio import Audio
from scipy.spatial.distance import cdist

class PyannoteDiarizer():
    def __init__(self, feature_clustering_threshold = 0.48, ema_alpha = 0.8, from_previous_session=None, max_num_speakers=10000):
        """
        :param device: device to run the model on
        :param feature_clustering_threshold: threshold for clustering the features
        :param ema_alpha: alpha for exponential moving average for the feature update
        """

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                             use_auth_token="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")
        
        self.diarization_pipeline.to(self.device)

        self.embedder = PretrainedSpeakerEmbedding("nvidia/speakerverification_en_titanet_large",
                                                   device=self.device)

        self.num_speakers = 0

        self.max_num_speakers = max_num_speakers

        self.ema_alpha = ema_alpha

        self.audio = Audio(sample_rate=16000, mono="downmix")
        
        if from_previous_session:
            self.diarization_pipeline.instantiate({'from': from_previous_session})

        self.features = {}

        self.feature_labels = [chr(i) for i in range(65, 91)] + [f"A{i}" for i in range(1, 1000)]

        self.feature_clustering_threshold = feature_clustering_threshold

    def __call__(self, kwargs: Any):
        """
        :param kwargs: dictionary with waveform and sample rate
        :return: Annotation object with speaker labels and segments

        Diarize the chunk and feature match the speakers with the previous chunks
        Determine if the speaker was already seen or not in the previous chunks
        """

        return self.run(kwargs["waveform"])

    def run(self, chunk: np.ndarray):
        """
        :param chunk: audio chunk
        :return: Annotation object with speaker labels and segments

        Diarize the chunk and feature match the speakers with the previous chunks
        Determine if the speaker was already seen or not in the previous chunks
        """

        diarization = self.get_diarization(chunk)
        embeddings = self.get_embeddings(chunk, diarization)
        segments = self.get_segments(diarization)
        speaker_labels = self.cluster_embeddings(embeddings)

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
        prev_embedding = None
        for i, embedding in enumerate(embeddings):
            if prev_embedding is not None and cdist(embedding, prev_embedding, metric="cosine")[0, 0] < self.feature_clustering_threshold:
                speaker_labels.append(speaker_labels[i - 1])
                self.features[speaker_labels[i - 1]] = [self.ema_alpha * self.features[speaker_labels[i - 1]][0] + (1 - self.ema_alpha) * embedding[0]]
                continue
            prev_embedding = embedding
            min_distance = np.inf
            distance = np.inf
            for key, previous_embedding in self.features.items():
                distance = cdist(embedding, previous_embedding, metric="cosine")[0, 0]
                if distance < min_distance:
                    min_distance = distance
                    candidate = key
            # print("\n \t min distance: ", round(min_distance, 2), end="\t \t")
            if min_distance < self.feature_clustering_threshold or self.num_speakers >= self.max_num_speakers:
                speaker_labels.append(candidate)
                # print(type(self.features[candidate]), end=" ")
                # print(len(self.features[candidate]))
                # print(type(embedding), end=" ")
                # print(len(embedding))
                # calculated_ema_alpha = self.ema_alpha + (((1 - self.ema_alpha) / self.feature_clustering_threshold) * min_distance)
                # print(round(calculated_ema_alpha, 2), end="\t \t")
                self.features[candidate] = [self.ema_alpha * self.features[candidate][0] + (1 - self.ema_alpha) * embedding[0]]
            else:
                self.features[self.feature_labels[self.num_speakers]] = embedding
                speaker_labels.append(self.feature_labels[self.num_speakers])
                self.num_speakers += 1

        return speaker_labels

        
    def get_diarization(self, chunk: np.ndarray):
        """
        :param chunk: audio chunk
        :return: diarization object
        """
        diarization = self.diarization_pipeline({'waveform': chunk, 'sample_rate': 16000})
        return diarization

    def get_segments(self, diarization):
        segments = []
        for track in diarization.itertracks(yield_label=True):
            if track[0].duration < 1:
                continue
            segments.append(track[0])
        return segments
    
    def segment_embedding(self, chunk: np.ndarray, segment: Segment):
        return self.embedder(chunk[:, int(segment.start * 16000):int(segment.end * 16000)].unsqueeze(1))
    
    def get_embeddings(self, chunk: np.ndarray, diarization):
        embeddings = []
        speaker_labels = []
        for track in diarization.itertracks(yield_label=True):
            segment = track[0]
            speaker = track[2]
            if segment.duration < 1:
                continue
            embeddings.append(self.segment_embedding(chunk, segment))
            speaker_labels.append(speaker)
        return embeddings
    
    def get_speaker_embeddings(self, diarization):
        embeddings = []
        for track in diarization.itertracks(yield_label=True):
            segment = track[0]
            speaker = track[2]
            embeddings.append((speaker, self.segment_embedding(segment)))
        return np.nan_to_num(embeddings)
    
    def reset(self):
        self.features = {}
        self.num_speakers = 0