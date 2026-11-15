"""Canonical API and registry data models (integration contract v1)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReviewStatus = Literal["pending", "approved", "rejected"]


class GlossEntry(ContractModel):
    gloss_id: str
    concept_fr: str
    gloss: str
    language: Literal["LSF"]
    variant: str
    source_id: str
    motion_id: str
    aliases: list[str]
    review_status: ReviewStatus


class MotionAsset(ContractModel):
    motion_id: str
    gloss_id: str
    path: str
    rig_version: Literal["signaculture-v1"]
    fps: float = Field(gt=0, allow_inf_nan=False)
    duration_ms: int = Field(gt=0)
    license: str
    quality_status: str
    rights_status: Literal["cleared", "pending"]
    animation_review: ReviewStatus
    synthetic: bool


class TranslationDraft(ContractModel):
    source_text: str
    tokens: list[str]
    lemmas: list[str]
    gloss_sequence: list[GlossEntry]
    unknowns: list[str]
    linguistic_review: Literal["pending"]
    translation_validated: Literal[False]


class AnimationSegment(ContractModel):
    gloss_id: str
    motion_id: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    asset_url: str


class AudioFrame(ContractModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    pitch_hz: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    energy: float = Field(ge=0, allow_inf_nan=False)
    is_pause: bool


    @model_validator(mode="after")
    def validate_interval(self):
        if self.start_ms >= self.end_ms:
            raise ValueError("Audio frame start_ms must precede end_ms")
        return self


class AudioFeatures(ContractModel):
    """Positive-duration audio; empty frames and null unvoiced pitch are valid."""
    duration_ms: int = Field(gt=0)
    frames: list[AudioFrame]
    emotion_hypothesis: None
    model_version: str
    warnings: list[str]


    @model_validator(mode="after")
    def validate_timeline(self):
        previous_end = 0
        for frame in self.frames:
            if frame.start_ms < previous_end:
                raise ValueError("Audio frames must be ordered and nonoverlapping")
            if frame.end_ms > self.duration_ms:
                raise ValueError("Audio frame exceeds audio duration_ms")
            previous_end = frame.end_ms
        return self


class AnimationPlan(ContractModel):
    plan_id: str
    source_text: str
    target_language: Literal["LSF"]
    segments: list[AnimationSegment]
    unknowns: list[str]
    duration_ms: int = Field(ge=0)
    linguistic_review: ReviewStatus
    animation_review: ReviewStatus
    rights_status: Literal["cleared", "pending"]
    render_allowed_for_publication: Literal[False]
    audio_features: AudioFeatures | None = None


class ReviewRecord(ContractModel):
    reviewer: str
    status: Literal["approved", "rejected"]
    notes: str
    version: str
    date: str
