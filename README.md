# DiffSynth-Studio Wan2.2-TI2V-5B Minimal Runtime

This repository has been reduced to the files needed to use the Wan2.2-TI2V-5B related DiffSynth code as a local folder import, without installing the full DiffSynth package.

## What remains

- `diffsynth/pipelines/wan_video.py`: Wan video pipeline, including the `WanVideoUnit_ImageEmbedderFused` path used by `Wan-AI/Wan2.2-TI2V-5B`.
- `diffsynth/models/wan_video_*.py`: Wan video model, text encoder, VAE, and related optional Wan components still imported by the Wan pipeline.
- `diffsynth/models/model_loader.py` and `diffsynth/configs/model_configs.py`: model loading and Wan model config metadata.
- `diffsynth/core`, `diffsynth/diffusion`, and selected `diffsynth/utils` modules: shared runtime helpers needed by the Wan pipeline.
- `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`: normal inference example.
- `examples/wanvideo/model_inference_low_vram/Wan2.2-TI2V-5B.py`: low-VRAM inference example.

Removed content includes unrelated image/audio pipelines, non-Wan model implementations, metrics, documentation, and non-TI2V examples/training scripts.

## Local import usage

Put this folder on `PYTHONPATH` or run your script from the repository root, then import directly:

```python
import torch
from diffsynth import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)
```

You can also use:

```python
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
```
