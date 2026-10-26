# Modified from https://github.com/and-mill/semantic-forgery

import torch
from .SD_pipe import SDPipe
from .PixArt_pipe import PixArtPipe
from diffusers import DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

# Map model id onto pipe. Add more if needed
DIFFUSION_PIPES = {
    'Manojb/stable-diffusion-2-1-base': SDPipe,
    'stable-diffusion-v1-5/stable-diffusion-v1-5': SDPipe,
    'PixArt-alpha/PixArt-XL-2-512x512': PixArtPipe,
}

SCHEDULERS = {
    "DDIM":DDIMScheduler,
    "DPM": DPMSolverMultistepScheduler
}

INVERSE_SCHEDULERS = {
    "DDIM": DDIMInverseScheduler,
    "DPM": DPMSolverMultistepInverseScheduler
}

def create_diffusion_pipe(pretrained_model_name_or_path: str,
                          resolution: int,
                          device: torch.device="cuda",
                          scheduler_name: str="DDIM",
                          scheduler_inverse_name: str="DDIM",
                          dtype: torch.dtype = torch.float32,
                          eager_loading: bool = False,
                          **kwargs):
    """
    API function call to create diffusion pipe
    @param pretrained_model_name_or_path: str
    @param resolution: int
    @param device: torch.device
    @param eager_loading: bool
    @param scheduler_name: str
    @param scheduler_inverse_name: str
    @param kwargs: dict

    @return: BaseDiffusionPipe
    """

    diffusion_pipe = DIFFUSION_PIPES[pretrained_model_name_or_path]
    return diffusion_pipe(pretrained_model_name_or_path=pretrained_model_name_or_path,
                          resolution=resolution,
                          device=device,
                          scheduler=SCHEDULERS[scheduler_name],
                          scheduler_inverse=INVERSE_SCHEDULERS[scheduler_inverse_name],
                          dtype=dtype,
                          eager_loading=eager_loading,
                          **kwargs)

    