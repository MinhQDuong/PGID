# Modified from https://github.com/and-mill/semantic-forgery - Thanks to the original authors for their work.

import typing
import torch
from abc import ABC, abstractmethod
from .image_utils import PIL_to_torch, torch_to_PIL
from diffusers import DDIMScheduler, DDIMInverseScheduler
from transformers import get_cosine_schedule_with_warmup
import PIL


class BaseDiffusionPipe(ABC):
    """
    Abstract base class for diffusion pipelines. Common interface for all diffusion pipelines.
    """
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 resolution: int,
                 device: torch.device,
                 scheduler: typing.Any = DDIMScheduler,
                 scheduler_inverse: typing.Any = DDIMInverseScheduler,
                 dtype: torch.dtype = torch.float32,
                 eager_loading: bool = False,
                 **kwargs):

        # Model id or path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # Prepare schedulers
        self.scheduler = scheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=dtype)
        self.scheduler_inverse = scheduler_inverse.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=dtype)

        self.resolution = resolution
        self.device = device

        # Eager loading of pipe
        self.kwargs = kwargs 
        self.dtype = dtype
        self.backbone = None
        self.pipe = self._get_pipe(self.pretrained_model_name_or_path, **self.kwargs) if eager_loading else None

    # Methods about latent dimensions
    @abstractmethod
    def get_num_latent_channels(self):
        """
        Return the number of latent channels in the diffusion pipeline.
        """
        raise NotImplementedError("get_num_latent_channels method must be implemented in diffusion pipelines.")

    def get_latent_resolution(self):
        self._load_pipe()
        return self.resolution // self.pipe.vae_scale_factor

    def get_image_shape(self, batch_size=1) -> typing.Tuple[int, int]:
        return (batch_size,
                3,
                self.resolution,
                self.resolution)

    def get_latent_shape(self, batch_size=1) -> typing.Tuple[int, int, int, int]:
        return (batch_size,
                self.get_num_latent_channels(),
                self.get_latent_resolution(),
                self.get_latent_resolution())


    @abstractmethod
    def get_diffusers_pipe_class(self):
        """
        Return the class of the diffusers pipeline.
        """
        raise NotImplementedError("get_diffusers_pipe_class method must be implemented in diffusion pipelines.")

    @abstractmethod
    def get_backbone(self):
        """
        Return the backbone of the diffusion pipeline. (Unet or DiT)
        """
        raise NotImplementedError("get_backbone method must be implemented in diffusion pipelines.")


    @abstractmethod
    def predict_noise_with_null(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Predicts noise with empty prompt using diffusion model
        Used later for inversion and null-text optimization
        """
        pass


    def _get_pipe(self, 
                  pretrained_model_name_or_path: str, 
                  disable_tqdm: bool = False, **kwargs):
        """
        Get the diffusion pipeline from the pretrained model name or path. This is used for lazy loading.
        @return: DiffusionPipeline object
        """
        pipe_class = self.get_diffusers_pipe_class()
        pipe = pipe_class.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          scheduler=self.scheduler,
                                          torch_dtype=self.get_dtype(),
                                          **kwargs)
        pipe.to(self.device)

        if disable_tqdm:
            pipe.set_progress_bar_config(disable=True)

        return pipe

    def _load_pipe(self):
        """
        Loads the diffusion pipeline if not already loaded. This is used for lazy loading
        """
        if self.pipe is None:
            self.pipe = self._get_pipe(self.pretrained_model_name_or_path, **self.kwargs)
        self.pipe = self.pipe.to(self.device)


    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes images into latent space using the VAE encoder.
        Accepts PIL, list of PIL, or
        torch tensor with or without batch dim, in [0, 1], in self.get_dtype()

        @param images: PIL, list of PIL, or torch tensor with or without batch dim, in [0, 1]
        @return: latents with batch dim on self.device
        """
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.PIL_to_torch(images)

        images = images.to(device=self.device, dtype=self.get_dtype())

        self._load_pipe()

        images = 2. * images - 1.
        with torch.no_grad():
            posterior = self.pipe.vae.encode(images).latent_dist
            latents = posterior.mean * self.pipe.vae.config.scaling_factor 
        latents = latents.to(self.device)

        return latents


    def imgs_to_latents(self, images: typing.Union[PIL.Image.Image,
                                                   typing.List[PIL.Image.Image],
                                                   torch.Tensor]) -> torch.Tensor:
        """
        Converts images to latent representations using vae_encode
        Accepts PIL, list of PIL, or
        torch tensor with or without batch dim, in [0, 1], in self.get_dtype()

        1. Transform to torch.Tensor if necessary
        2. Put image to self.device
        3. get latent z0 with self.pipe.vae

        @param images: PIL, list of PIL, or torch tensor with or without batch dim, in [0, 1]
        @return latents: latents with batch dim on self.device
        """

        # to torch if necessary
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.PIL_to_torch(images).to(self.device)
        
        images = images.to(self.device)

        # unsqueeze
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        # push through vae
        latents = self.vae_encode(images).detach()

        return latents


    def latents_to_imgs(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Accepts latents with batch dim on self.device

        @param latents: latents with batch dim on self.device
        @return: images with batch dim on self.device - already divided by scaling_factor
        """
        self._load_pipe()

        latents = latents.to(device=self.device,dtype=self.get_dtype())

        images = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor # Rescale back
        ).sample.detach()

        return images

    def generate(self,
                 prompts: typing.List[str],
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 latents: typing.Optional[torch.Tensor] = None,
                 num_images_per_prompt: int = 1,
                 return_latents: bool = True) -> dict:
        """
        Generate dict with

        @param prompt: str
        @param num_inference_steps: int
        @param guidance_scale: float
        @param latents: torch.Tensor with batch dim
        @param num_images_per_prompt: int
        @param return_latents: bool

        @return dict
        """
        self._load_pipe()

        self.pipe.scheduler = self.scheduler

        # make sure latents are on device
        if latents is not None:
            latents = latents.to(self.device, dtype=self.get_dtype())

        # generate
        out = self.pipe(
             prompts,
             num_images_per_prompt=num_images_per_prompt,
             guidance_scale=guidance_scale,
             num_inference_steps=num_inference_steps,
             height=self.resolution,
             width=self.resolution,
             latents=latents,
        )
    
        # collect results
        images_PIL = out.images # list of PIL image
        images_torch = self.PIL_to_torch(out.images)

        if return_latents:
            z0_torch = self.imgs_to_latents(out.images)
            z0_PIL = torch_to_PIL(z0_torch)

            zT_torch = latents.to(self.device) if latents is not None else torch.zeros(z0_torch.shape).to(self.device)
            zT_PIL = torch_to_PIL(latents) if latents is not None else torch.zeros(z0_torch.shape).to(self.device)
        
        return {
            # images
            'images_torch': images_torch,
            'images_PIL': images_PIL,
            'images': images_PIL,
            # z0
            'z0_torch': z0_torch if return_latents else None,
            'z0_PIL': z0_PIL if return_latents else None,
            'z0': z0_PIL if return_latents else None,
            # zT
            'zT_torch': zT_torch if return_latents else None,
            'zT_PIL': zT_PIL if return_latents else None,
            'zT': zT_PIL if return_latents else None,}

    def invert_z0(self,
                  latents: torch.Tensor,
                  num_inference_steps: int = 50) -> torch.tensor:
        """
        Do DDIM inversion on given latents z0

        @param latents: torch tensor with batch dim
        @param num_inference_steps: int

        @return: zT torch tensor with batch dim on self.device
        """
        self._load_pipe()
        latents = latents.to(device=self.device, dtype=self.get_dtype())
        # set to inverse scheduler
        self.pipe.scheduler = self.scheduler_inverse

        # invert z0 to zT
        try:
            out = self.pipe(latents=latents,
                            num_inference_steps=num_inference_steps,
                            prompt=[""] * latents.shape[0],
                            negative_prompt="",
                            guidance_scale=1.,
                            width=self.resolution,
                            height=self.resolution,
                            output_type='latent',
                            return_dict=False,
                        )[0]
        finally:
            # reset back to default scheduler
            self.pipe.scheduler = self.scheduler
        return out

    def invert_images(self,
                      images: typing.Union[PIL.Image.Image,
                                           typing.List[PIL.Image.Image],
                                           torch.Tensor] = None,
                      num_inference_steps: int = 50,
                      skip_inversion: bool = False,
                      latents: torch.Tensor = None):
        """
        Do DDIM inversion on given images
        
        Accepts PIL, list of PIL, or
        torch tensor with or without batch dim, in [0, 1]

        @param images: PIL, list of PIL, or torch tensor with or without batch dim, in [0, 1]
        @param num_inference_steps: int
        @param skip_inversion: bool
        @param latents: latents

        @return: dict
        """

        assert images is not None or latents is not None, "Either images or latents must be given"

        # cast to torch on device
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.PIL_to_torch(images)

        # collect results
        z0_torch = self.imgs_to_latents(images) if latents is None else latents
        z0_PIL = torch_to_PIL(z0_torch)

        zT_torch = self.invert_z0(z0_torch, num_inference_steps=num_inference_steps) if not skip_inversion else z0_torch
        zT_PIL = torch_to_PIL(zT_torch)
        
        return {
            'z0_torch': z0_torch,
            'z0_PIL': z0_PIL,
            'zT_torch': zT_torch,
            'zT_PIL': zT_PIL
        }

    # Decoder inversion from PRC
    def decoder_inv(self, x, num_steps=100, init_z0=None):
        """
        Optimization to find z such that Decode(z) ~ x.
        @param x: image data
        @param num_steps: number of optimization steps
        @param init_z0: initial latent data, if None, will use init guess
        @return z: modifed latent data
        """
        self._load_pipe()
        original_dtype = self.pipe.vae.dtype
        self.pipe.vae.to(torch.float32)
        
        self.pipe.vae.eval()
        self.pipe.vae.requires_grad_(False)

        # Start with the init guess
        if init_z0 is not None:
            z = init_z0.clone().to(torch.float32)
        else:
            with torch.no_grad():
                z = self.pipe.vae.encode(x.to(torch.float32)).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        
        z.requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=0.1)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=num_steps)
        
        target_img = x.clone().float()
        scaling_factor = self.pipe.vae.config.scaling_factor
        
        for _ in range(num_steps):
            rec_img = self.pipe.vae.decode(z / scaling_factor).sample  
            
            loss = torch.nn.functional.mse_loss(rec_img, target_img, reduction='sum')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
        self.pipe.vae.to(original_dtype)
        return z.detach()

    # Exact inversion for PRC - https://arxiv.org/abs/2311.18387
    def exact_inversion(self, images, num_inference_steps: int = 50, decode_inv=True) -> torch.Tensor:
        """
        Performs exact inversion for PRC - work similarly to invert_images
        with optional decoder inversion step. This is used by PRC for watermark detection
        @param images: 
        @param num_inference_steps: number of inference steps
        @param decode_inv: whether to perform decoder inversion
        @return dict with zT and z0
        """
        z0 = self.imgs_to_latents(images)

        if decode_inv:
            # Prepare the target image in the [-1, 1]
            if isinstance(images, PIL.Image.Image) or isinstance(images, list):
                image_pt = self.PIL_to_torch(images).to(self.device)
            else:
                image_pt = images.to(self.device)
                
            if image_pt.ndim == 3:
                image_pt = image_pt.unsqueeze(0)
                
            image_pt_scaled = 2.0 * image_pt - 1.0
            
            z0 = self.decoder_inv(image_pt_scaled, num_steps=100)
            
        zT = self.invert_z0(latents=z0, num_inference_steps=num_inference_steps)

        # collect results
        z0_PIL = torch_to_PIL(z0)
        zT_PIL = torch_to_PIL(zT)

        return {
            'z0_torch': z0,
            'z0_PIL': z0_PIL,
            'zT_torch': zT,
            'zT_PIL': zT_PIL
        }

    # Common utils functions for all diffusion pipelines
    def get_dtype(self):
        """
        Returns the data type of the model.
        """
        return self.dtype

    def stash_pipe(self):
        """Drop pipe to free up memory"""
        if self.pipe is not None:
            self.pipe = self.pipe.to(torch.device("cpu"))
        if self.backbone is not None:
            self.backbone = self.backbone.to(torch.device("cpu"))

    def PIL_to_torch(self, images: typing.Union[PIL.Image.Image, typing.List[PIL.Image.Image]], dtype=None) -> torch.Tensor:
        """
        Accepts PIL, list of PIL,
        One or more images to torch tensor with batch dim
        
        @param images: PIL, list of PIL
        @return latents: latents with batch dim on self.device
        """
        dtype = self.get_dtype() if dtype is None else dtype    
        return PIL_to_torch(images, dtype=dtype, device=self.device)