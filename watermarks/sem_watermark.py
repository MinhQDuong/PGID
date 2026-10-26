# Modified from https://github.com/and-mill/semantic-forgery

from tqdm import tqdm
import typing
import torch


class SemanticWatermark():
    def __init__(self,
                 latent_shape: typing.Tuple[int, int, int, int],
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cuda'),
                 **kwargs):

        """
        Base class to config different watermarking schemes. A Watermark object defines a watermark signal

        @param latent_shape: Shape of the latent tensor with batch dim either tuple or torch.shape. e.g. (1, 4, 64, 64)
        @param dtype: device
        @param device: device
        """

        if isinstance(latent_shape, tuple):
            latent_shape = torch.Size(latent_shape)
            self.latent_shape = latent_shape
            self.batch_size = latent_shape[0] # The number of images that would carry this watermark signal
            self.num_channels = latent_shape[1]
            self.latent_resolution = latent_shape[2]

        assert len(latent_shape) == 4 and latent_shape[-1] == latent_shape[-2]

        self.dtype = dtype
        self.device = device


    @classmethod
    def generate_watermark_signals(cls,
                           total_num_signals: int,
                           latent_shape: typing.Tuple[int, int, int, int],
                           w_seed: int = None,
                           target_start_index: int = 0,
                           target_end_index: int = None,
                           **provider_args: dict) -> typing.Generator:

        """
        Returns a generator which consists of different watermark signals for each latent batch.
        

        @param total_num_signals: Total number of watermark signals to generate.
        @param latent_shape: Shape of the latent tensor with batch dim . e.g. (1, 4, 64, 64)
        @param w_seed: Seed for the watermark generation.
        @param target_start_index: Start index of the range of latents to generate watermarks for.
        @param target_end_index: End index of the range of latents to generate watermarks for.
        @param provider_args: Additional arguments to pass to the watermark provider.

        """

        if target_end_index is None:
            target_end_index = total_num_signals
        else:
            target_end_index = min(target_end_index, total_num_signals)

        for idx in tqdm(range(target_start_index, target_end_index)):

            # Here offset is used to get different keys, nonces and messages for GS.
            # Get different messages for PRC
            # Get different master keys, session keys and messages for T2SMark
            # Different offset will result in different watermark signals except for Tree-Ring
            # where by design the same signal is used for all latents.
   
            offset = idx # Using idx as offset to get different watermark signals for each latent

            watermark = cls(
                latent_shape=latent_shape,
                offset=offset,
                w_seed=w_seed, # Same seed gives same TR signal
                **provider_args,
            )

            yield watermark, idx


    