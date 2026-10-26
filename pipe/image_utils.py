# Originally from https://github.com/and-mill/semantic-forgery

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import typing
import os


def load_pil(filename: str, dir_name: str = "cache"):
    """
    load a PIL image from a file
    """

    # Full path for loading the image
    full_path = os.path.join(dir_name, filename)
    
    # Load and return the image
    return Image.open(full_path)


def save_pil(image: Image.Image, filename: str, dir_name: str = "cache"):
    """
    Save a PIL image to a file
    """
    
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Full path for saving the image
    full_path = os.path.join(dir_name, filename)
    
    image.save(full_path)
    
    # Load the image to verify
    loaded_image = load_pil(filename)
    
    # Verify by comparing the two images
    assert list(image.getdata()) == list(loaded_image.getdata()), "Image not saved correctly"

def scale_tensor_to_range(tensor: torch.Tensor,
                          min_val: float = 0.0,
                          max_val: float = 1.0) -> torch.Tensor:
    """
    Scale a tensor to a given range.

    @param tensor: tensor to scale
    @param min_val: minimum value of the range
    @param max_val: maximum value of the range

    @return tensor: scaled tensor
    """
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * (max_val - min_val) + min_val

    return tensor


def torch_to_PIL(images: typing.Union[torch.Tensor, np.ndarray],
                 scale_to_pixel_vals: bool = True) -> typing.List[Image.Image]:
    """
    Images will be scaled to [0, 1]. All images in batch will be considered for determining this range.
    
    @param images: torch tensor with or without batch dim in [0, 1]. Also allows numpies, will be casted immediately
    @param scale_to_pixel_vals: bool, if True, will scale to [0, 255] and cast to uint8

    @return images: list of PIL images
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

      # SD3
    if images.shape[1] == 16:
            # Option 1: Average across the 16 latent channels to reduce to 3 channels
            images = images.mean(dim=1, keepdim=True)  # Mean over the 16 channels, keep dimension
            images = images.expand(-1, 3, -1, -1)  # Expand to 3 channels

    if scale_to_pixel_vals:
        images = scale_tensor_to_range(images, 0, 1)
        images = images * 255
        images = images.to(torch.uint8)
    images = images.detach().cpu()

    # Ensure the input is 4D (batch, channel, height, width) or 3D (channel, height, width)
    if images.dim() not in [3, 4]:
        raise ValueError("Input tensor must be 3D or 4D")

    # Prepare to convert each image in the batch
    if images.dim() == 4:
        # Batch of multi-channel images
        # colormap chosen automatically here. 1 channel -> grayscale, 3 channels -> RGB, 4 channels -> RGBA, more channels -> not enforced
        return [transforms.functional.to_pil_image(img) for img in images]
    else:
        # Batch of greyscale images
        # colormap chosen automatically here. 1 channel -> grayscale, 3 channels -> RGB, 4 channels -> RGBA, more channels -> not enforced
        return [transforms.functional.to_pil_image(img) for img in images]


def PIL_to_torch(images: typing.Union[Image.Image,
                                      typing.List[Image.Image]],
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Accepts PIL, list of PIL,
    One or more images to torch tensor with batch dim

    @param images: PIL, list of PIL
    @param dtype: dtype
    @param device: device

    @return latents: latents with batch dim on cpu
    """
    transform = transforms.ToTensor()

    if isinstance(images, Image.Image):
        images = transform(images)
    elif isinstance(images, list):
        images = torch.stack([transform(i) for i in images])

    return images.to(dtype=dtype, device=device)