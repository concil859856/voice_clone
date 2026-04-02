from typing import Optional

from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str = Field(description="Transcript of the reference audio")


class VoiceCloneResponse(BaseModel):
    reference_transcript: str = Field(description="ASR text used as ref_text for voice cloning")
    transcript_source: str = Field(description="'provided' | 'asr'")
    text_spoken: str = Field(description="Target text that was synthesized")
    sample_rate: int
    audio_base64: str = Field(description="WAV PCM output, base64-encoded")
