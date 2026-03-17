# import librosa


# def load_audio(audio_file):

#     audio, sr = librosa.load(audio_file, sr=16000)

#     return audio


# def segment_audio(audio, sr=16000, window_sec=6):

#     segment_length = window_sec * sr

#     segments = []

#     for i in range(0, len(audio), segment_length):
#         segments.append(audio[i:i+segment_length])

#     return segments

import librosa

FRAME_SIZE = 0.5   # seconds
STRIDE = 0.25      # seconds


def load_audio(audio_file):

    audio, sr = librosa.load(audio_file, sr=16000)

    return audio, sr


def split_frames(audio, sr):

    frame_len = int(FRAME_SIZE * sr)
    stride_len = int(STRIDE * sr)

    frames = []
    timestamps = []

    for start in range(0, len(audio) - frame_len, stride_len):

        end = start + frame_len

        frames.append(audio[start:end])

        timestamps.append(start / sr)

    return frames, timestamps