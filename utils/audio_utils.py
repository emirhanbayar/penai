import numpy as np
import librosa
import os
from scipy.io import wavfile
import random
import time
import torch

class AudioIterator():
    """
        Takes an audio file and streams it in batches to the models as if the transcrition was done in real time
    """
    def __init__(self, audio_path, chunk_size, sr=16000):
        """
        :param audio_path: path to audio file
        :param batch_size: batch size
        :param sr: sampling rate
        :param chunk_size: size of each chunk
        """
        self.sr = sr
        self.chunk_size = chunk_size
        # read wav file 
        self.audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        # 2D array with shape (channels, samples)
        self.audio = np.expand_dims(self.audio, axis=0)
        self.audio = torch.from_numpy(self.audio).float()
        self.audio_length = self.audio.shape[1]
        self.num_batches = int(np.ceil(self.audio_length / float(self.chunk_size)))
        self.batch_pointer = 0
        print(f"Audio duration: {self.audio_length / float(self.sr)}")
        print(f"Number of batches: {self.num_batches}")

    def next(self, simulate_real_time=False):
        """
        :return: next batch of audio
        """

        if simulate_real_time:
            # simulate real time by randomly waiting between 0 and 1 seconds
            wait_time = random.random()
            time.sleep(wait_time)

        if self.batch_pointer < self.num_batches:
            batch_start = self.batch_pointer * self.chunk_size
            batch_end = min(self.audio_length, batch_start + self.chunk_size)
            batch = self.audio[:, batch_start:batch_end]
            print(type(batch))
            self.batch_pointer += 1
            return batch, batch_start / float(self.sr), batch_end / float(self.sr)
        else:
            return None, None, None
        
    def reset(self):
        """
        reset batch pointer
        """
        self.batch_pointer = 0

def np_to_wav(batch, path, sample_rate=16000):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    wavfile.write(path, sample_rate, np.array(batch))

    # return duration of wav file in seconds
    return len(batch) / float(sample_rate)
