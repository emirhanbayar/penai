import argparse
import os
from pyannote.database.util import load_rttm
from pyannote.core import notebook, Annotation, Segment
from pyannote.metrics.diarization import DiarizationPurity, DiarizationCoverage, DiarizationErrorRate
import matplotlib.pyplot as plt

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
    audio_files = [os.path.join(audio_data_path, f) for f in os.listdir(audio_data_path)]
    rttm_files = [os.path.join(rttm_data_path, f) for f in os.listdir(rttm_data_path) if f.endswith(".rttm")]
    audio_files.sort()
    rttm_files.sort()
    return audio_files, rttm_files

if __name__ == "__main__":
    with open("data/aishell-4/test/TextGrid/L_R003S01C02.rttm") as f:
        reference = load_rttm(f)["L_R003S01C02"]
    # visualize the diarization
    with open("output/aishell-4/test/pyannote-ours_100000000000000000_0.8/L_R003S01C02.rttm") as f:
        hypothesis = load_rttm(f)["L_R003S01C02"]
    with open("output/aishell-4/test/pyannote-vanilla/L_R003S01C02.rttm") as f:
        vanilla = load_rttm(f)["L_R003S01C02"]
    seg_start = 0
    seg_end = 1000
    crop = Segment(seg_start, seg_end)
    notebook.crop = crop
    notebook.start_time = seg_start
    notebook.duration = seg_end - seg_start
    plt.subplot(311)
    notebook.plot_annotation(reference, legend=True)
    plt.subplot(312)
    notebook.reset()
    notebook.crop = crop
    notebook.start_time = seg_start
    notebook.duration = seg_end - seg_start
    notebook.plot_annotation(hypothesis, legend=True)
    plt.subplot(313)
    notebook.reset()
    notebook.crop = crop
    notebook.start_time = seg_start
    notebook.duration = seg_end - seg_start
    notebook.plot_annotation(vanilla, legend=True)
    plt.show()
    # compute DER between 0 and 100 seconds
    reference = reference.crop(crop)
    hypothesis = hypothesis.crop(crop)
    metric = DiarizationPurity()
    print("Purity = ", metric(reference, hypothesis))
    metric = DiarizationCoverage()
    print("Coverage = ", metric(reference, hypothesis))
    metric = DiarizationErrorRate()
    print("DER = ", metric(reference, hypothesis))