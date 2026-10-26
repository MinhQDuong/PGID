
import torch
from diffusers import PixArtAlphaPipeline, PixArtTransformer2DModel
from .base_pipe import BaseDiffusionPipe

# Hardcode the number of latent channels for PixArt to 4
NUM_LATENT_CHANNELS = 4

class PixArtPipe(BaseDiffusionPipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override so PixArt uses leading timestep spacing (the default of SD)
        self.scheduler.register_to_config(
            timestep_spacing="leading"
        )
        self.scheduler_inverse.register_to_config(
            timestep_spacing="leading"
        )

    def get_diffusers_pipe_class(self):
        """
        Return the diffusers pipeline class.
        """
        return PixArtAlphaPipeline

    def get_backbone(self):
        """
        Return the backbone of the diffusion pipeline. (Unet or DiT)

        """
        if self.backbone is None:
            if self.pipe is not None:
                # Reuse the transformer from the loaded pipeline
                self.backbone = self.pipe.transformer
            else:
                # Load only the transformer for lazy loading
                self.backbone = PixArtTransformer2DModel.from_pretrained(
                    self.pretrained_model_name_or_path,
                    subfolder="transformer",
                    torch_dtype=self.get_dtype()
                )
                self.backbone = self.backbone.to(self.device)

        return self.backbone

    def get_num_latent_channels(self):
        return NUM_LATENT_CHANNELS

    def prepare_null_prompt_embeds(self, latents: torch.Tensor) -> dict:
        """
        Get the null prompt embeddings.
        @param latents: torch.Tensor, the latents to be used for the null prompt embeddings (have batch size)
        @return: dict 
        """
        self._load_pipe()
        batch_size = latents.shape[0]
        with torch.no_grad():
            outputs = self.pipe.encode_prompt(
                prompt=[""] * batch_size,
                do_classifier_free_guidance=False
            )
        prompt_embeds = outputs[0]
        prompt_attention_mask = outputs[1]

        added_cond_kwargs = {
            "resolution": torch.tensor(
                [[self.resolution, self.resolution]] * batch_size,
                device=self.device,
                dtype=self.get_dtype(),
            ),
            "aspect_ratio": torch.ones(
                (batch_size, 1),
                device=self.device,
                dtype=self.get_dtype(),
            ),
        }

        return {
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_attention_mask,
            "added_cond_kwargs": added_cond_kwargs,
        }

    # Null_embeds is now a dict containing encoder_hidden_states, encoder_attention_mask, and added_cond_kwargs
    def predict_noise_with_null(self, latents: torch.Tensor, timestep: torch.Tensor, null_embeds: dict) -> torch.Tensor:
        """
        Predicts noise with empty prompt using diffusion model (DiT)
        """
        self.get_backbone()

        batch_size = latents.shape[0]

        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)

        noise_pred_full = self.backbone(
            hidden_states=latents,
            timestep=timestep,
            **null_embeds
        )[0]
        # PixArt outputs both noise and variance, keep only the noise (first 4 channels)
        noise_pred, _ = noise_pred_full.chunk(2, dim=1)
        return noise_pred