from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from .base_pipe import BaseDiffusionPipe
import torch

# Hardcode the number of latent channels for Stable Diffusion to 4
NUM_LATENT_CHANNELS = 4

class SDPipe(BaseDiffusionPipe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_diffusers_pipe_class(self):
        """
        Return the diffusers pipeline class.
        """
        return StableDiffusionPipeline


    def get_backbone(self):
        """
        Return the UNet backbone, loading it lazily if needed.
        """
        if self.backbone is None:
            if self.pipe is not None:
                # Reuse the UNet from the loaded pipeline
                self.backbone = self.pipe.unet
            else:
                # Load only the UNet for lazy loading
                self.backbone = UNet2DConditionModel.from_pretrained(
                    self.pretrained_model_name_or_path,
                    subfolder="unet",
                    torch_dtype=self.get_dtype()
                )
                self.backbone = self.backbone.to(self.device)

        return self.backbone


    def get_num_latent_channels(self):
        return NUM_LATENT_CHANNELS

    def prepare_null_prompt_embeds(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Get the null prompt embeddings.
        @param latents: torch.Tensor, the latents to be used for the null prompt embeddings (have batch size)
        @return: torch.Tensor, the null prompt embeddings
        """
        self._load_pipe()
        batch_size = latents.shape[0]
        with torch.no_grad():
            null_embeds, _ = self.pipe.encode_prompt(
                prompt=[""] * batch_size, device=self.pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
        return null_embeds


    def predict_noise_with_null(self, latents: torch.Tensor, timestep: torch.Tensor, null_embeds: torch.Tensor) -> torch.Tensor:
        """
        Predicts noise with empty prompt using diffusion model (Unet)
        """
        self.get_backbone()
        noise_pred = self.backbone(
                        sample=latents,
                        timestep=timestep,
                        encoder_hidden_states=null_embeds,
                        return_dict=False,
                    )[0]
        return noise_pred



    



    