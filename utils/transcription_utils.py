def concatenate_segments(segments):
    """
    Concatenate the segments from faster-whisper into a single string
    """
    transcription_text = ""
    for segment in segments:
        transcription_text += segment.text
    return transcription_text


def jsonify_word(word):
    """
    Convert a faster-whisper word object into a JSON object
    """
    return {
        "word": word.word,
        "start": word.start,
        "end": word.end,
        "probability": word.probability,
        "tokens": None,
    }


def jsonify_segment(segment):
    """
    Convert a faster-whisper segment object into a JSON object
    """
    return {
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob,
        "id": segment.id,
        "words": [jsonify_word(word) for word in segment.words],
    }


def format_transcription(segments, info):
    """
    Format the transcription from faster-whisper into the required format for stable-ts inference
    """
    language = info.language
    text = concatenate_segments(segments)
    try:
        words = [jsonify_word(word) for segment in segments for word in segment.words]
    except:
        words = []
    segments = [jsonify_segment(segment) for segment in segments]

    return {"language": language, "text": text, "segments": segments, "words": words}
