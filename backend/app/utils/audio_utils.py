import librosa


def load_audio(audio_file):

    audio, sr = librosa.load(audio_file, sr=16000)

    return audio


def segment_audio(audio, sr=16000, window_sec=6):

    segment_length = window_sec * sr

    segments = []

    for i in range(0, len(audio), segment_length):
        segments.append(audio[i:i+segment_length])

    return segments