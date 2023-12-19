import argparse
import os

from diarization.pyannote_diarizer import PyannoteDiarizer
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation, notebook
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationPurity, DiarizationCoverage, DiarizationErrorRate
import matplotlib.pyplot as plt

from utils.audio_utils import AudioIterator

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset-name", type=str, default="aishell-4", help="Name of dataset")
    parser.add_argument("--dataset-type", type=str, default="test", help="Type of dataset")
    parser.add_argument("--dataset-size", type=str, default="small", help="Size of dataset")
    parser.add_argument("--dataset-path", type=str, default="data", help="Path to store dataset")
    parser.add_argument("--diarizer", type=str, default="pyannote-ours", help="Diarization model")
    parser.add_argument("--output-path", type=str, default="output", help="Path to store dataset")
    parser.add_argument("--feature-clustering-threshold", type=float, default=0.8, help="Threshold for clustering features")
    parser.add_argument("--ema-alpha", type=float, default=0.90, help="Feature update")
    parser.add_argument("--chunk-size", type=int, default=64000, help="Chunk size for diarization")
    parser.add_argument("--generate-rttm", action="store_true", help="Generate rttm files")

    return parser.parse_args()

def download_dataset(dataset_name, dataset_type, dataset_size, dataset_path):
    filename = None
    if dataset_name == "aishell-4":
        if dataset_type == "train":
            filename = f"train_{dataset_size}.tar.gz"
            os.system(f"wget https://us.openslr.org/resources/111/train_{dataset_size}.tar.gz")
        elif dataset_type == "test":
            filename = "test.tar.gz"
            os.system("wget https://us.openslr.org/resources/111/test.tar.gz")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    
    os.makedirs(f"{dataset_path}/{dataset_name}", exist_ok=True)
    os.system(f"tar -xvf {filename} -C {dataset_path}/{dataset_name}")
    os.system(f"rm {filename}")

    return os.path.join("data", dataset_name, dataset_type)

def create_diarizer(diarizer_name, args):
    if diarizer_name == "pyannote-ours":
        return PyannoteDiarizer(feature_clustering_threshold=args.feature_clustering_threshold, ema_alpha=args.ema_alpha)
    elif diarizer_name == "pyannote-vanilla":
        return Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    else:
        raise NotImplementedError(f"Diarizer {diarizer_name} not implemented")
    
def load_data(dataset_path, dataset_name, dataset_type, dataset_size):

    audio_files = []
    rttm_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                audio_files.append(os.path.join(root, file))
            elif file.endswith(".rttm"):
                rttm_files.append(os.path.join(root, file))

    audio_files = sorted(audio_files)
    rttm_files = sorted(rttm_files)
    return audio_files, rttm_files

def load_rttms(args, desired_name=None):
    outputs = []
    ground_truths = []
    if desired_name is None:
        desired_name = f"{args.diarizer}_{args.chunk_size}_{args.feature_clustering_threshold}"

    for root, _, files in os.walk(f"{args.output_path}/{args.dataset_name}/{args.dataset_type}/{desired_name}"):
        for file in files:
            if file.endswith(".rttm"):
                print(os.path.join(root, file))
                outputs.append(os.path.join(root, file))

    for root, _, files in os.walk(f"data/{args.dataset_name}/{args.dataset_type}"):
        for file in files:
            if file.endswith(".rttm"):
                ground_truths.append(os.path.join(root, file))

    outputs = sorted(outputs)
    ground_truths = sorted(ground_truths)

    hyps = []
    refs = []

    for output, ground_truth in zip(outputs, ground_truths):
        try:
            hyps.append(load_rttm(output)[output.split('/')[-1].split('.')[0]])
            refs.append(load_rttm(ground_truth)[ground_truth.split('/')[-1].split('.')[0]])
        except:
            print(f"Could not load {output} or {ground_truth}")

    return hyps, refs

def evaluate(hyps, refs, mid_refs=None):
    num_results = min(len(hyps), len(refs))
    print(f"Evaluating {num_results} files")
    coverage = DiarizationCoverage()
    purity = DiarizationPurity()
    error_rate = DiarizationErrorRate()
    for i in range(num_results):
        coverage(refs[i], hyps[i])
        purity(refs[i], hyps[i])
        error_rate(refs[i], hyps[i])
    print(f"Coverage: {coverage.report()}")
    print(f"Purity: {purity.report()}")
    print(f"Error rate: {error_rate.report()}")

    if mid_refs is not None:
        mid_coverage = DiarizationCoverage()
        mid_purity = DiarizationPurity()
        mid_error_rate = DiarizationErrorRate()
        for i in range(num_results):
            mid_coverage(refs[i], mid_refs[i])
            mid_purity(refs[i], mid_refs[i])
            mid_error_rate(refs[i], mid_refs[i])
        print(f"Mid Coverage: {mid_coverage.report()}")
        print(f"Mid Purity: {mid_purity.report()}")
        print(f"Mid Error rate: {mid_error_rate.report()}")

def visualize(hyps, refs, mid_refs=None, args=None):
    if args is not None:
        if mid_refs is not None:
            num_results = min(len(hyps), len(refs), len(mid_refs))
            title_1 = f"{args.dataset_name} {args.dataset_type} Reference"
            title_2 = f"{args.diarizer} {args.chunk_size} {args.feature_clustering_threshold}"
            title_3 = f"pyannote-vanilla"
    num_results = min(len(hyps), len(refs))
    print(f"Visualizing {num_results} files")
    for i in range(num_results):
        seg_start = 0
        seg_end = 100
        crop = Segment(seg_start, seg_end)
        # full screen
        plt.figure(figsize=(20, 10))
        plt.subplot(211)
        plt.text(80, 0.8, title_1)
        notebook.reset()
        notebook.crop = crop
        notebook.start_time = seg_start
        notebook.duration = seg_end - seg_start
        notebook.plot_annotation(refs[i], legend=True)
        plt.subplot(212)
        plt.text(80, 0.8, title_2)
        notebook.reset()
        notebook.crop = crop
        notebook.start_time = seg_start
        notebook.duration = seg_end - seg_start
        notebook.plot_annotation(hyps[i], legend=True)
        if mid_refs is not None:
            plt.subplot(313)
            plt.text(80, 0.8, title_3)
            notebook.reset()
            notebook.crop = crop
            notebook.start_time = seg_start
            notebook.duration = seg_end - seg_start
            notebook.plot_annotation(mid_refs[i], legend=True)

        plt.tight_layout()
        
        os.makedirs(f"plots/{args.dataset_name}/{args.dataset_type}/{args.diarizer}_{args.chunk_size}_{args.feature_clustering_threshold}", exist_ok=True)
        plt.savefig(f"plots/{args.dataset_name}/{args.dataset_type}/{args.diarizer}_{args.chunk_size}_{args.feature_clustering_threshold}/{i}.png")

def main():
    args = parse_args()
    if os.path.exists(f"{args.dataset_path}/{args.dataset_name}/{args.dataset_type}"):
        print(f"Dataset {args.dataset_name} already exists")
    else:
        print(f"Downloading dataset {args.dataset_name}...")
        download_dataset(args.dataset_name, args.dataset_type, args.dataset_size, args.dataset_path)

    if args.generate_rttm:
        diarizer = create_diarizer(args.diarizer, args)
        audio_files, rttm_files = load_data(args.dataset_path, args.dataset_name, args.dataset_type, args.dataset_size)
        for audio_file, rttm_file in zip(audio_files, rttm_files):
            if args.diarizer == "pyannote-vanilla":
                chunk_size = np.inf
            else:
                chunk_size = args.chunk_size
            audio_iterator = AudioIterator(audio_file, chunk_size)
            if args.diarizer == "pyannote-ours":
                os.makedirs(f"{args.output_path}/{args.dataset_name}/{args.dataset_type}/{args.diarizer}_{chunk_size}_{args.feature_clustering_threshold}", exist_ok=True)
                f = open(f"{args.output_path}/{args.dataset_name}/{args.dataset_type}/{args.diarizer}_{chunk_size}_{args.feature_clustering_threshold}/{os.path.basename(audio_file).split('.')[0]}.rttm", "w")
            else:
                os.makedirs(f"{args.output_path}/{args.dataset_name}/{args.dataset_type}/{args.diarizer}", exist_ok=True)
                f = open(f"{args.output_path}/{args.dataset_name}/{args.dataset_type}/{args.diarizer}/{os.path.basename(audio_file).split('.')[0]}.rttm", "w")
    
            result = Annotation(uri=os.path.basename(audio_file).split('.')[0])
            while True:
                chunk, global_start, global_end = audio_iterator.next(simulate_real_time=True)
                if chunk is None:
                    break
                try:
                    speaker_labels, segments = diarizer({'waveform': chunk, 'sample_rate': 16000})
                except:
                    print(f"Could not diarize chunk: {global_start} - {global_end}")
                for i, segment in enumerate(segments):
                    result[Segment(start=global_start + segment.start, end=global_start + segment.end)] = speaker_labels[i]
            result.write_rttm(f)

    hyps, refs = load_rttms(args)
    mid_refs, _ = load_rttms(args, "pyannote-vanilla")
    
    evaluate(hyps, refs, mid_refs)

    visualize(hyps, refs, mid_refs, args)


if __name__ == "__main__":
    main()