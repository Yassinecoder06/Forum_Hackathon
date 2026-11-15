"""Canonical API and registry data models (integration contract v1)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    fps: float = Field(gt=0)
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
    linguistic_review: ReviewStatus
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
    pitch_hz: float | None
    energy: float = Field(ge=0)
    is_pause: bool


class AudioFeatures(ContractModel):
    duration_ms: int = Field(gt=0)
    frames: list[AudioFrame]
    emotion_hypothesis: None
    model_version: str
    warnings: list[str]


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
