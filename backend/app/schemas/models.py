"""Canonical API and registry data models (integration contract v1)."""

from typing import Annotated, Literal, Optional

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
    # Token index range [start, end) each gloss was matched from, parallel to gloss_sequence.
    gloss_token_spans: list[tuple[int, int]] | None = None


class AnimationSegment(ContractModel):
    gloss_id: str
    motion_id: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    asset_url: str
    # Part of the source clip to play (its signing core); absent = whole clip.
    clip_start_ms: Optional[int] = Field(default=None, ge=0)
    clip_end_ms: Optional[int] = Field(default=None, gt=0)
    # Voice prosody for this sign (timing_mode "audio_words"); acoustic estimates, not annotation.
    emphasis: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    liveliness: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    hold_ms: Optional[int] = Field(default=None, ge=0, le=1000)
    word_start_ms: Optional[int] = Field(default=None, ge=0)
    word_end_ms: Optional[int] = Field(default=None, gt=0)


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


EmotionScores = dict[str, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]


class EmotionWindow(ContractModel):
    """One overlapping emotion2vec window; scores are model hypotheses."""
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    scores: EmotionScores

    @model_validator(mode="after")
    def validate_interval(self):
        if self.start_ms >= self.end_ms:
            raise ValueError("Emotion window start_ms must precede end_ms")
        return self


class ProsodyWord(ContractModel):
    text: str = Field(max_length=200)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    prominence: float = Field(allow_inf_nan=False)
    emphasized: bool
    syl_per_s: float = Field(ge=0, allow_inf_nan=False)
    liveliness: float = Field(ge=0, le=1, allow_inf_nan=False)


class Prosody(ContractModel):
    """Word-level acoustic prominence, pace and pitch liveliness (estimates, not annotation)."""
    pace_syl_per_s: float = Field(ge=0, allow_inf_nan=False)
    words: list[ProsodyWord]


class AudioFeatures(ContractModel):
    """Positive-duration audio; empty frames and null unvoiced pitch are valid."""
    duration_ms: int = Field(gt=0)
    frames: list[AudioFrame]
    emotion_hypothesis: EmotionScores | None
    emotion_timeline: list[EmotionWindow] | None = None
    model_version: str
    warnings: list[str]
    transcription: "AudioTranscript | None" = None
    prosody: Prosody | None = None


    @model_validator(mode="after")
    def validate_timeline(self):
        previous_end = 0
        for frame in self.frames:
            if frame.start_ms < previous_end:
                raise ValueError("Audio frames must be ordered and nonoverlapping")
            if frame.end_ms > self.duration_ms:
                raise ValueError("Audio frame exceeds audio duration_ms")
            previous_end = frame.end_ms
        previous_start = -1
        for window in self.emotion_timeline or []:
            if window.start_ms <= previous_start:
                raise ValueError("Emotion windows must be ordered by start_ms")
            if window.end_ms > self.duration_ms:
                raise ValueError("Emotion window exceeds audio duration_ms")
            previous_start = window.start_ms
        if self.transcription:
            for segment in self.transcription.segments:
                if segment.end_ms > self.duration_ms:
                    raise ValueError("Transcript segment exceeds audio duration_ms")
        return self


class TranscriptWord(ContractModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(max_length=200)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class TranscriptSegment(ContractModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(max_length=2000)
    words: list[TranscriptWord] | None = None

    @model_validator(mode="after")
    def validate_interval(self):
        if self.start_ms >= self.end_ms:
            raise ValueError("Transcript segment start_ms must precede end_ms")
        return self


class AudioTranscript(ContractModel):
    text: str = Field(max_length=10000)
    language: Literal["fr"]
    model_id: str
    model_revision: str
    device: Literal["cuda", "cpu"]
    compute_type: str
    segments: list[TranscriptSegment]


AudioFeatures.model_rebuild()


class AnimationPlan(ContractModel):
    transition_ms: int = Field(default=0,ge=0,le=1000)
    timing_mode: Literal["original", "audio_voiced", "audio_words"] = "original"
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
