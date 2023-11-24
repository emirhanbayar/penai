# PENAI - DIARIZATION & TRANSCRIPTION

This branch contains a collections of pluggable modules for Audio Signal Processing, Diarization and Speech Recognition.

## Installation
It is recommended to use a virtual environment for the installation.
```bash
conda create --name penai python=3.8
conda activate penai
```
Install the requirements
```bash
pip install -r requirements.txt
```

## Usage
The following script will print out the diarized text of the audio file.

```bash
python main.py --audio_file <path_to_audio_file> --chunk_size <chunk_size> --feature_clustering_threshold <threshold> --from_previous_session <path_to_previous_session> --device <device>
```

The default values for the arguments are as follows:

| Argument | Default Value |
| --- | --- |
| audio_file | an4_diarize_test.wav |
| chunk_size | 32000 |
| feature_clustering_threshold | 0.8 |
| from_previous_session | None |
| device | cpu |
