import torch
import librosa

from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-er"
)

model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er"
)

model.to(device)

emotion_labels = model.config.id2label


def segment_audio(audio, sr=16000, window_sec=6):

    segment_length = window_sec * sr

    segments = []

    for i in range(0, len(audio), segment_length):
        segments.append(audio[i:i+segment_length])

    return segments


def predict_emotion(segment):

    inputs = feature_extractor(
        segment,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred = torch.argmax(logits).item()

    return emotion_labels[pred]


def emotion_timeline(audio_file):

    audio, sr = librosa.load(audio_file, sr=16000)

    segments = segment_audio(audio)

    timeline = []

    start = 0

    for seg in segments:

        emotion = predict_emotion(seg)

        duration = len(seg)/16000

        end = start + duration

        timeline.append({
            "start": round(start,2),
            "end": round(end,2),
            "emotion": emotion
        })

        start = end

    return timeline