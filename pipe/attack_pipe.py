# Originally from https://github.com/and-mill/semantic-forgery
# Pipe used by attacker, here only supports Stable Diffusion as attacker

from .pipe_utils import create_diffusion_pipe
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Union
import PIL
import os

def create_attack_pipe(
        pretrained_model_name_or_path: str = "Manojb/stable-diffusion-2-1-base",
        resolution: int = 512,
        device: str = "cuda",
        scheduler_name: str = "DDIM",
        scheduler_inverse_name: str = "DDIM",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
    """
    API function call to create attack pipe (default to eager loading)
    @param pretrained_model_name_or_path: str 
    @param resolution: int
    @param device: torch.device
    @param scheduler_name: str
    @param scheduler_inverse_name: str
    @param kwargs: dict

    @return: (StableDiffusionPipeline, Scheduler, InverseScheduler)
    """
    attack_pipe_wrapper = create_diffusion_pipe(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                 resolution=resolution,
                                 device=device,
                                 scheduler_name=scheduler_name,
                                 scheduler_inverse_name=scheduler_inverse_name,
                                 dtype=dtype,
                                 eager_loading=True,
                                 **kwargs)

    attack_pipe = attack_pipe_wrapper.pipe
    attack_pipe.set_progress_bar_config(disable=True)
    scheduler = attack_pipe_wrapper.scheduler
    inverse_scheduler = attack_pipe_wrapper.scheduler_inverse

    return attack_pipe, scheduler, inverse_scheduler


class DiffPipe(torch.nn.Module):
    """
    A class for backpropagating through a pipeline's generation loop using gradient checkpointing.
    Works for Stable Diffusion
    """
    def __init__(self, pipe, scheduler=None, device=None):
        super().__init__()
        self.pipe = pipe
        self.device = device
        self.scheduler = scheduler
        
        self.unet = UnetWrapper(self.pipe.unet)
        
        for param in self.pipe.unet.parameters():
            param.requires_grad = False
        """
        Forward pass through the pipeline with gradient checkpointing.

        @param latents: torch.Tensor, the latents
        @param prompt: str, the prompt
        @param negative_prompt: str, the negative prompt
        @param prompt_embeds: torch.Tensor, the prompt embeddings
        @param guidance_scale: float, the guidance scale
        @param num_inference_steps: int, the number of inference steps
        @param device: torch.device, the device
        @param do_classifier_free_guidance: bool, whether to do classifier free guidance
        @param generator: torch.Generator, the generator
        @param eta: float, the eta
        """
        
    def forward(self,   
                latents=None,
                prompt="", 
                negative_prompt="",
                prompt_embeds=None,
                guidance_scale=7.5,
                num_inference_steps: int = 50,
                device=None,
                do_classifier_free_guidance=None,
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                eta: float = 0.0,
                ):
        """
        Forward pass through the pipeline with gradient checkpointing.

        @param latents: torch.Tensor, the latents
        @param prompt: str, the prompt
        @param negative_prompt: str, the negative prompt
        @param prompt_embeds: torch.Tensor, the prompt embeddings
        @param guidance_scale: float, the guidance scale
        @param num_inference_steps: int, the number of inference steps
        @param device: torch.device, the device
        @param do_classifier_free_guidance: bool, whether to do classifier free guidance
        @param generator: torch.Generator, the generator
        @param eta: float, the eta
        """
        oldsched = self.pipe.scheduler
        self.pipe.scheduler = self.scheduler
        
        sigmas = None
        batch_size = 1
        num_images_per_prompt = 1
        timesteps = None
        do_classifier_free_guidance = (guidance_scale > 1.) if do_classifier_free_guidance is None else do_classifier_free_guidance
        
        device = device if device is not None else self.device
        
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale > 0,
                negative_prompt
            )
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds = prompt_embeds
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        latents = latents * self.scheduler.init_noise_sigma

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = None

        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipe._num_timesteps = len(timesteps)
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = checkpoint(self.unet, 
                    latent_model_input,
                    t,
                    prompt_embeds,
                    timestep_cond,
                    self.pipe.cross_attention_kwargs,
                    added_cond_kwargs,
                    False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self.pipe.scheduler = oldsched
        return latents
            
    
class UnetWrapper(torch.nn.Module):
    """ Necessary to wrap Unet because gradient checkpoint doesn't allow kwargs so need to make used args a tuple. """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        
    def forward(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    timestep_cond,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict,):
        ret = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,)
        return ret
    
    
def latent_to_pil(latents, pipe) -> PIL.Image:
    """
    Simple utils for converting a latent to an image

    @param latents: torch.Tensor
    @param pipe: DiffusionPipeline (likely StableDiffusionPipeline)

    @return: PIL.Image
    """
    with torch.no_grad():
        image = pipe.vae.decode(latents.detach() / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
    ret = [to_pil_image((img * 0.5 + 0.5).clamp(0, 1)) for img in image]
    return ret
    
    
def pixel_to_latent(image, pipe):
    """
    Simple utils for converting an image to a latent

    @param image: PIL.Image or torch.Tensor
    @param pipe: DiffusionPipeline
    """
    if not isinstance(image, torch.Tensor):
        image = to_tensor(image).to(pipe.device)[None]
        image = 2. * image - 1.
    
    posterior = pipe.vae.encode(image).latent_dist
    z0 = posterior.mean
    z0 = z0.to(pipe.device)
    z0 = z0 * pipe.vae.config.scaling_factor 
    
    return z0
    

def invert_image_with_attack_pipe(pipe=None, scheduler=None,
                 image=None,
                 image_pt=None,
                 num_inference_steps=50,
                 guidance_scale=1,
                 resolution=512
                 ) -> torch.Tensor:
    """
    Simple util for inverting using the attack pipe. Returns zT (latent at timestep T) for the given image.

    @param pipe: DiffusionPipeline (likely StableDiffusionPipeline)
    @param image: PIL.Image or torch.Tensor
    @param image_pt: torch.Tensor
    @param num_inference_steps: int
    @param guidance_scale: float
    @param resolution: int

    @return: torch.Tensor
    """
    if image_pt is None:
        image_pt = to_tensor(image).to(pipe.device)[None]
        image_pt = 2. * image_pt - 1.
    else:
        assert image is None
        
    posterior = pipe.vae.encode(image_pt).latent_dist
    z0 = posterior.mean
    z0 = z0.to(pipe.device)
    z0 = z0 * pipe.vae.config.scaling_factor 

    # set to inverse scheduler
    orig_sched = pipe.scheduler
    pipe.scheduler = scheduler

    try:
        # invert z0 to zT
        zT_retrieved = pipe(latents=z0,
                            num_inference_steps=num_inference_steps,
                            prompt=[""] * z0.shape[0],
                            negative_prompt=[""],
                            guidance_scale=guidance_scale,
                            width=resolution,
                            height=resolution,
                            output_type='latent',
                            return_dict=False,
                            )[0]
    finally:
        pipe.scheduler = orig_sched
    return zT_retrieved