from utils.audio_utils import AudioIterator


from diarization.pyannote_diarizer import PyannoteDiarizer
import argparse
from transcription.whisper_transcriber import WhisperSpeechTranscriber
from transcription.model_config import ModelSize
import time

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import time
import soundfile as sf

class Denoiser():
    def _init_(self, device = None, model = None):
        self.model = pretrained.dns64().cpu()

    def denoise(self, file, chunk_counter):
    
        start = time.time()
        print("starting to denoise")
    
        wav, sr = torchaudio.load(file)
        wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

        with torch.no_grad():
            denoised = self.model(wav[None])[0]
        print(f"it took {time.time() - start} seconds to get the output")

        return denoised.cpu(), self.model.sample_rate  # Return the result on the CPU if necessary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=str, default="emirhan.wav")
    parser.add_argument("--chunk-size", type=int, default=64000)
    parser.add_argument("--feature-clustering-threshold", type=float, default=0.7)
    parser.add_argument("--from-previous-session", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-num-speakers", type=int, default=7)


    return parser.parse_args()


if __name__ == "__main__":
    # initialize audio iterator
    args = parse_args()
    audio_iterator = AudioIterator(args.audio_path, args.chunk_size)

    denoise_model = pretrained.dns64().cuda()

    # initialize diarizer
    diarizer = PyannoteDiarizer(device=args.device,
                                feature_clustering_threshold=args.feature_clustering_threshold,
                                from_previous_session=args.from_previous_session,
                                max_num_speakers=args.max_num_speakers)

    # initialize diarizer
    diarizer = PyannoteDiarizer(
        device=args.device,
        feature_clustering_threshold=args.feature_clustering_threshold,
        from_previous_session=args.from_previous_session,
    )

    # initialize speech transcriber
    transcriber = WhisperSpeechTranscriber(ModelSize.LARGE_V2)
    # iterate over audio
    start_time = time.time()
    start_time = time.time()
    while True:
        chunk, global_start, global_end = audio_iterator.next(simulate_real_time=True)
        if chunk is None:
            break
        
        wav = convert_audio(chunk.cuda(), 16000, denoise_model.sample_rate, denoise_model.chin)

        with torch.no_grad():
            chunk = denoise_model(wav[None])[0]

        # chunk = chunk.cpu()

        speaker_labels, segments = diarizer.run(chunk)

        text, words = transcriber.get_transcription(chunk[0, :])
        print(f"\n \n Chunk {audio_iterator.batch_pointer}", end="\t")

        # wait for the time that is left
        total_time = global_end - (time.time() - start_time)
        if total_time > 0:
            print(
                "Completed ", global_end - (time.time() - start_time), " seconds early"
            )
            time.sleep(global_end - (time.time() - start_time))
        else:
            print(
                "Completed ",
                abs(global_end - (time.time() - start_time)),
                " seconds late !!!!!",
            )

        for i, segment in enumerate(segments):
            print()
            print(
                f"\t Speaker {speaker_labels[i]}: {round(global_start + segment.start,3)} - {round(global_start + segment.end,3)}",
                end="\t \t",
            )
            print("\t Text: ", end="")
            try:
                speaker_text = ""
                for word in words:
                    if (
                        segment.start <= word["start"]
                        and word["end"] <= segment.end
                        and word["word"] != "nd"
                    ):
                        speaker_text += word["word"] + " "
            except:
                print("")
        # print remaining time segments
        try:
            print(speaker_text,end="")
        except:
            pass

    print()
