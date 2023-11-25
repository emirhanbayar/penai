import argparse
import os

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from utils.dataset_utils import download_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="aishell-4")
    parser.add_argument("--dataset-type", type=str, default="test")
    parser.add_argument("--dataset-size", type=str, default="")
    parser.add_argument("--audio-data-path", type=str, default="./data/aishell-4/test/wav")
    parser.add_argument("--rttm-data-path", type=str, default="./data/aishell-4/test/TextGrid")
    parser.add_argument("--chunk-size", type=int, default=32000)
    parser.add_argument("--feature-clustering-threshold", type=float, default=0.8)
    parser.add_argument("--run-scratch", action="store_true")
    parser.add_argument("--device", type=str, default="gpu")

    return parser.parse_args()

def load_data(audio_data_path, rttm_data_path):
    audio_files = [os.path.join(audio_data_path, f) for f in os.listdir(audio_data_path) if f.endswith(".flac")]
    rttm_files = [os.path.join(rttm_data_path, f) for f in os.listdir(rttm_data_path) if f.endswith(".rttm")]
    audio_files.sort()
    rttm_files.sort()
    return audio_files, rttm_files

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.audio_data_path):
        download_dataset(args.dataset_name, args.dataset_type, args.dataset_size)

    audio_files, rttm_files = load_data(args.audio_data_path, args.rttm_data_path)
    if args.run_scratch:
        for i in range(len(audio_files)):
            print(f"Processing {audio_files[i]}")
            os.system(f"python main.py --audio-path {audio_files[i]} --chunk-size {args.chunk_size} --feature-clustering-threshold {args.feature_clustering_threshold} --device {args.device} --save-rttm {os.path.basename(audio_files[i]).replace('.wav', '.rttm')}")
        