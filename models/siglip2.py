import torch.nn as nn
import torch.nn.functional as F


class SigLIP2Encoder(nn.Module):
    def __init__(
        self,
        name="google/siglip2-large-patch16-512",
        image_size=224,
        local_files_only=False,
    ):
        super().__init__()
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:
            raise ImportError(
                "SigLIP2Encoder requires `transformers` with SiglipVisionModel support."
            ) from exc

        self.name = "siglip2_large_patch16_512"
        self.source_name = name
        self.image_size = image_size
        self.base_model = SiglipVisionModel.from_pretrained(
            name,
            local_files_only=local_files_only,
        )
        self.emb_dim = self.base_model.config.hidden_size
        self.latent_ndim = 2
        self.patch_size = self.base_model.config.patch_size

    def forward(self, x):
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        outputs = self.base_model(pixel_values=x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state
