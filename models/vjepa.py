import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPAEncoder(nn.Module):
    def __init__(
        self,
        name="facebook/vjepa2-vitl-fpc64-256",
        image_size=224,
        input_frames=2,
        local_files_only=False,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, VJEPA2Model
        except ImportError as exc:
            raise ImportError(
                "VJEPAEncoder requires `transformers` with VJEPA2Model support."
            ) from exc

        self.name = "vjepa2_vitl"
        self.source_name = name
        self.image_size = image_size
        self.input_frames = input_frames
        self._cfg = AutoConfig.from_pretrained(name, local_files_only=local_files_only)
        self.base_model = VJEPA2Model.from_pretrained(
            name,
            local_files_only=local_files_only,
        )
        self.emb_dim = self._cfg.hidden_size
        self.latent_ndim = 2
        self.patch_size = self._cfg.patch_size

    def forward(self, x):
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        video = x.unsqueeze(1).repeat(1, self.input_frames, 1, 1, 1)
        outputs = self.base_model(pixel_values_videos=video, skip_predictor=True)
        return outputs.last_hidden_state
