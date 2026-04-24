# import torch

# from app.models.model_loader import (
#     model,
#     feature_extractor,
#     emotion_labels,
#     device
# )

# from app.utils.audio_utils import load_audio, segment_audio


# def predict_emotion(segment):

#     inputs = feature_extractor(
#         segment,
#         sampling_rate=16000,
#         return_tensors="pt"
#     )

#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():

#         logits = model(**inputs).logits

#     pred = torch.argmax(logits).item()

#     return emotion_labels[pred]


# def emotion_timeline(audio_file):

#     audio = load_audio(audio_file)

#     segments = segment_audio(audio)

#     timeline = []

#     start = 0

#     for seg in segments:

#         emotion = predict_emotion(seg)

#         duration = len(seg) / 16000

#         end = start + duration

#         timeline.append({
#             "start": round(start,2),
#             "end": round(end,2),
#             "emotion": emotion
#         })

#         start = end

#     return timeline

import torch

from app.models.model_loader import (
    model,
    feature_extractor,
    emotion_labels,
    device
)

from app.utils.audio_utils import (
    load_audio,
    split_frames
)


def predict_frames(frames):

    emotions = []

    for frame in frames:

        inputs = feature_extractor(
            frame,
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():

            logits = model(**inputs).logits

        pred = torch.argmax(logits).item()

        emotions.append(emotion_labels[pred])

    return emotions

def smooth_predictions(emotions, timestamps, min_duration=0.5):

    smoothed_emotions = emotions.copy()
    smoothed_timestamps = timestamps.copy()

    # Step 1: Majority voting (window size = 3)
    for i in range(1, len(emotions) - 1):

        window = [emotions[i - 1], emotions[i], emotions[i + 1]]

        majority = max(set(window), key=window.count)

        smoothed_emotions[i] = majority

    # Step 2: Remove very short segments
    final_emotions = []
    final_timestamps = []

    i = 0
    while i < len(smoothed_emotions):

        current_emotion = smoothed_emotions[i]
        start_time = smoothed_timestamps[i]

        j = i + 1
        while j < len(smoothed_emotions) and smoothed_emotions[j] == current_emotion:
            j += 1

        end_time = smoothed_timestamps[j - 1]

        duration = end_time - start_time

        # If segment too short → merge with previous
        if duration < min_duration and len(final_emotions) > 0:

            final_emotions[-1] = final_emotions[-1]  # keep previous emotion

        else:
            final_emotions.append(current_emotion)
            final_timestamps.append(start_time)

        i = j

    return final_emotions, final_timestamps


def detect_transitions(timestamps, emotions):

    timeline = []

    start_time = timestamps[0]
    current_emotion = emotions[0]

    for i in range(1, len(emotions)):

        if emotions[i] != current_emotion:

            end_time = timestamps[i]

            timeline.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "emotion": current_emotion
            })

            start_time = timestamps[i]
            current_emotion = emotions[i]

    timeline.append({
        "start": round(start_time, 2),
        "end": round(timestamps[-1], 2),
        "emotion": current_emotion
    })

    return timeline


def emotion_timeline(audio_file):

    audio, sr = load_audio(audio_file)

    frames, timestamps = split_frames(audio, sr)

    emotions = predict_frames(frames)

    smoothed_emotions, smoothed_timestamps = smooth_predictions(emotions, timestamps)

    timeline = detect_transitions(smoothed_timestamps, smoothed_emotions)

    return timeline