from pyannote.audio import Pipeline
import os
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook


diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

diarizer.to(torch.device("cuda"))

def get_ground_truth(dataset_name, dataset_type):
    audio_files = []
    rttm_files = []
    for root, dirs, files in os.walk(f"data/aishell-4/test/wav"):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                audio_files.append(os.path.join(root, file))
            elif file.endswith(".rttm"):
                rttm_files.append(os.path.join(root, file))

    audio_files = sorted(audio_files)
    rttm_files = sorted(rttm_files)
    return audio_files, rttm_files

def main():
    audio_files, rttm_files = get_ground_truth("whisper", "test")
    # run diarizer on each audio file and save the output
    for audio_file in audio_files:
        print(f"Diarizing {audio_file}")
        with ProgressHook() as hook:
            diarizer(audio_file, hook=hook)
        diarizer.write_rttm(f"{audio_file}.rttm")
        
if __name__ == "__main__":
    main()