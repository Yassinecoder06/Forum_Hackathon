"""Local inference for the SDPose-Wholebody release."""

from collections import OrderedDict
import cv2
import numpy as np


def heatmap_head():
    from torch import nn

    # The SDPose fork uses parameter-free InstanceNorm2d followed by SiLU.
    # The standard MMPose BatchNorm/ReLU head is a different architecture.
    return nn.Sequential(OrderedDict([
        ("deconv_layers", nn.Sequential(
            nn.ConvTranspose2d(640, 640, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(640),
            nn.SiLU(inplace=True),
        )),
        ("conv_layers", nn.Sequential(
            nn.Conv2d(640, 640, 1),
            nn.InstanceNorm2d(640),
            nn.SiLU(inplace=True),
        )),
        ("final_layer", nn.Conv2d(640, 133, 1)),
    ]))


def predictor(checkpoint, device, precision, decode):
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTokenizer

    kind = torch.bfloat16 if precision == "bf16" else torch.float32
    options = {"torch_dtype": kind, "local_files_only": True, "use_safetensors": True}
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("SDPose BF16 requires a GPU with native BF16 support")

    tokenizer = CLIPTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(checkpoint / "text_encoder", **options).to(device).eval()
    with torch.inference_mode():
        token_ids = tokenizer("", padding=False, truncation=True, return_tensors="pt").input_ids.to(device)
        text_embedding = text_encoder(token_ids).last_hidden_state
    del text_encoder

    vae = AutoencoderKL.from_pretrained(checkpoint / "vae", **options).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(checkpoint / "unet", **options).to(device).eval()
    head = heatmap_head()
    state = load_file(checkpoint / "decoder" / "decoder.safetensors")
    head.load_state_dict(state, strict=True)
    head = head.to(device=device, dtype=kind).eval()
    timestep = torch.tensor(999, device=device, dtype=torch.long)
    task = torch.tensor([[1.0, 0.0]], device=device)
    task = torch.cat((task.sin(), task.cos()), dim=-1).to(kind)
    width, height = 768, 1024

    class PoseFeatures(Exception):
        def __init__(self, features):
            self.features = features

    def capture_features(_module, _inputs, output):
        # Wholebody uses the penultimate up block. Later blocks do not feed the head.
        raise PoseFeatures(output)

    unet.up_blocks[-2].register_forward_hook(capture_features)
    generator = torch.Generator(device=device)

    @torch.inference_mode()
    def predict(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = np.array(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
        tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float32).permute(2, 0, 1)
        tensor = (tensor.unsqueeze(0) / 127.5 - 1).to(kind)
        distribution = vae.encode(tensor).latent_dist
        latent = distribution.sample(generator=generator.manual_seed(0)) * 0.18215
        try:
            unet(latent, timestep, text_embedding, class_labels=task, return_dict=False)
        except PoseFeatures as output:
            features = output.features
        else:
            raise RuntimeError("The SDPose UNet did not produce Wholebody features")
        heatmaps = head(features)[0].float().cpu().numpy()
        if heatmaps.shape != (133, height // 4, width // 4):
            raise ValueError(f"Unexpected SDPose heatmap shape: {heatmaps.shape}")
        if not np.isfinite(heatmaps).all():
            raise ValueError("SDPose returned non-finite heatmaps")
        points, scores = decode(heatmaps, (width, height), (width // 4, height // 4))
        source_height, source_width = image.shape[:2]
        points *= np.array([source_width / width, source_height / height])
        return points[0], scores[0]

    return predict
