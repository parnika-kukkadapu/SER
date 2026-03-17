from fastapi import APIRouter, UploadFile
import shutil

from app.services.emotion_service import emotion_timeline
from app.schemas.response import EmotionTimelineResponse

router = APIRouter()


@router.post("/predict", response_model=EmotionTimelineResponse)

async def predict(file: UploadFile):

    path = "temp.wav"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    timeline = emotion_timeline(path)

    return {"timeline": timeline}