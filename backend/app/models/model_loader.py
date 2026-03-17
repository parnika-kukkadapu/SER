import torch
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading emotion model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-er"
)

model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er"
)

model.to(device)
model.eval()

emotion_labels = model.config.id2label

print("Model loaded successfully.")