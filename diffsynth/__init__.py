"""Minimal DiffSynth package focused on Wan2.2-TI2V-5B inference."""

from .pipelines.wan_video import WanVideoPipeline, ModelConfig

__all__ = ["WanVideoPipeline", "ModelConfig"]
