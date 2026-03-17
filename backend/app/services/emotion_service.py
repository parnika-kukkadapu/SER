import torch

from app.models.model_loader import (
    model,
    feature_extractor,
    emotion_labels,
    device
)

from app.utils.audio_utils import load_audio, segment_audio


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

    audio = load_audio(audio_file)

    segments = segment_audio(audio)

    timeline = []

    start = 0

    for seg in segments:

        emotion = predict_emotion(seg)

        duration = len(seg) / 16000

        end = start + duration

        timeline.append({
            "start": round(start,2),
            "end": round(end,2),
            "emotion": emotion
        })

        start = end

    return timeline