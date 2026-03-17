from pydantic import BaseModel
from typing import List


class EmotionSegment(BaseModel):
    start: float
    end: float
    emotion: str


class EmotionTimelineResponse(BaseModel):
    timeline: List[EmotionSegment]