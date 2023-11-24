from utils.audio_utils import AudioIterator


from diarization.pyannote_diarizer import PyannoteDiarizer
import argparse
from transcription.whisper_transcriber import WhisperSpeechTranscriber
from transcription.model_config import ModelSize
import time

import torch
torch.cuda.empty_cache()

import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=str, default="an4_diarize_test.wav")
    parser.add_argument("--chunk-size", type=int, default=32000)
    parser.add_argument("--feature-clustering-threshold", type=float, default=0.8)
    parser.add_argument("--from-previous-session", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-diar-result", action="store_true")


    return parser.parse_args()


if __name__ == "__main__":
    # initialize audio iterator
    args = parse_args()
    audio_iterator = AudioIterator(args.audio_path, args.chunk_size)

    # initialize diarizer
    diarizer = PyannoteDiarizer(device=args.device,
                                feature_clustering_threshold=args.feature_clustering_threshold,
                                from_previous_session=args.from_previous_session)


    # initialize speech transcriber
    transcriber=WhisperSpeechTranscriber(ModelSize.LARGE_V2)
    # iterate over audio
    json_dict = {}
    start_time = time.time()
    start_time = time.time()
    while True:
        chunk, global_start, global_end = audio_iterator.next(simulate_real_time=True)
        if chunk is None:
            break
        speaker_labels, segments = diarizer.run(chunk)

        text=transcriber.get_transcription(chunk[0, :])
        print(f"\n \n Chunk {audio_iterator.batch_pointer}", end="\t")

        # wait for the time that is left
        total_time = global_end - (time.time() - start_time)
        if total_time > 0:
            print("Completed ", global_end - (time.time() - start_time), " seconds early")
            time.sleep(global_end - (time.time() - start_time))
        else:
            print("Completed ", abs(global_end - (time.time() - start_time)), " seconds late !!!!!")

        for i, segment in enumerate(segments):
            json_dict[global_start + segment.start] = {"speaker": speaker_labels[i], "text": text["segments"][i%len(text["segments"])]["text"]}
            print()
            print(f"\t Speaker {speaker_labels[i]}: {round(global_start + segment.start,3)} - {round(global_start + segment.end,3)}", end="\t \t")
            print(f"\t Text: ", end="")
            try:
                print(text["segments"][i%len(text["segments"])]["text"], end="")
            except:
                print("")
        # print remaining time segments
        if i < len(segments) - 1:
            for j in range(i + 1, len(segments)):
                print(text["segments"][j%len(text["segments"])]["text"], end="")

    with open("result.json", "w") as f:
        json.dump(json_dict, f)

    print()
