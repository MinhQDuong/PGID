# Adapted from https://github.com/0xD009/T2SMark

import torch
from functools import reduce
from scipy.stats import norm
from .sem_watermark import SemanticWatermark
from PIL import Image
from pipe.image_utils import torch_to_PIL
import numpy as np
import typing
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--message_width_in_bytes', default=32, type=int, help="Number of bytes in the watermark payload")
parser.add_argument('--tau', default=0.674, type=float, help="Safety threshold for Tail-Truncated Sampling")
parser.add_argument('--key_length', default=16, type=int, help="Length of the master key")
parser.add_argument('--key_channel_idx', default=0, type=int, help="The channel index used for encoding the key")

class T2SMarkWatermark(SemanticWatermark):
    def __init__(self,
                 message_width_in_bytes: int = 32,  # (channels/f_c) * f_h * f_w // 8
                 tau: float = 0.674, # According to the paper
                 key_length: int = 16,
                 offset: int = 0,
                 key_channel_idx: int = 0,
                 message: typing.Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.message_width_in_bytes = message_width_in_bytes
        self.message_width_in_bits = message_width_in_bytes * 8
        self.tau = tau
        self.prng = torch.Generator(device=self.device)
        self.key_length = key_length
        self.key_channel_idx = key_channel_idx
        self.offset = offset

        # Master key for each watermark signal
        master_key_gen = torch.Generator(device=self.device)
        master_key_gen.manual_seed(2 * self.offset) # Master keys use even seeds

        # Generate the key using the isolated generator
        self.master_key = torch.randint(0, 2, (self.key_length,), generator=master_key_gen, device=self.device)
        self.msg_channel_idx = [i for i in range(self.num_channels) if i != self.key_channel_idx] # num_channels = 4

        # If message not given, use from the list, indexing using offset
        if message is None:
            from .messages import MESSAGES
            MESSAGES = [m[:self.message_width_in_bytes] for m in MESSAGES]  # trim to message_width_in_bytes to fit our required message length
            self.message = MESSAGES[offset]
        else:
            self.message = message

    def get_wm_type(self) -> str:
        return "T2SMark"

    def __binlist2int(self, binlist):
        res = reduce(lambda x, y: x * 2 + y, binlist)
        if isinstance(binlist, torch.Tensor):
            return res.item()
        return res


    def __encode(self, b, K, target_shape):
        """
            T2SMark encoding function
        """
        n = reduce(lambda x, y: x * y, target_shape, 1) # Total space available
        m = len(b)                                      # Length of the payload
        r = int(2 * norm.cdf(-self.tau) * n / m)        # Replication factor
        k = m * r                                       # Total message bits
        noise_size = n - k                              # Remaining space for noise

        z = torch.randn(target_shape, device=self.device).flatten()
        self.prng.manual_seed(self.__binlist2int(K))
        
        v_value = torch.randint(0, 2, (k,), generator=self.prng, device=self.device).float() * 2 - 1
        v_support = torch.randperm(n, generator=self.prng, device=self.device)[:k]

        b_r = (1 - 2 * b).repeat(r).to(self.device).float()
        codeword = b_r * v_value

        w = torch.zeros(n, dtype=torch.bool, device=self.device)
        w[v_support] = True

        tail = torch.topk(z.abs(), k=k, dim=0, largest=True, sorted=False)
        central = torch.topk(z.abs(), k=noise_size, dim=0, largest=False, sorted=False)

        z_w = torch.zeros(n, device=self.device)
        z_w[w] = tail.values * codeword
        z_w[~w] = central.values * (torch.randint(0, 2, (noise_size,), device=self.device).float() * 2 - 1)
        
        return z_w.reshape(target_shape)


    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        """
        Get Watermarked latents - T2SMark encoding
        The same master key, session key and message for a watermark signal
                        
        @return: dict
        """

        batch_size = self.batch_size # The number of images that would carry this watermark signal
        
        # get the message
        message_bytes = self.message 

        # Construct messages
        message_bits_str = ''.join(format(byte, '08b') for byte in message_bytes) 
        message_bits_str_list = [message_bits_str] * batch_size

        # Get the message in bits
        message_bits = torch.tensor([int(b) for b in message_bits_str])
        session_key_gen = torch.Generator(device=self.device)
        session_key_gen.manual_seed(2 * self.offset + 1)
        session_key = torch.randint(0, 2, (self.key_length,), generator=session_key_gen, device=self.device)

        latents_torch = []
        for _ in range(batch_size):
            z_k = self.__encode(session_key, self.master_key, [1, self.latent_resolution, self.latent_resolution])
            z_b = self.__encode(message_bits, session_key, [self.num_channels - 1, self.latent_resolution, self.latent_resolution])

            latent_torch = torch.zeros(1, self.num_channels, self.latent_resolution, self.latent_resolution, device=self.device)
            latent_torch[:, self.key_channel_idx, :, :] = z_k
            latent_torch[:, self.msg_channel_idx, :, :] = z_b

            latents_torch.append(latent_torch)

        # finalize
        latents_torch = torch.cat(latents_torch, dim=0)
        latents_PIL = torch_to_PIL(latents_torch)

        # Clean latent for comparing maybe
        latents_clean = torch.randn_like(latents_torch, dtype=self.dtype, device=self.device)
        latents_clean_PIL = torch_to_PIL(latents_clean)

        results_dict = {"zT_torch": latents_torch,
                        "zT_clean_torch": latents_clean,
                        "zT_PIL": latents_PIL,
                        "zT_clean_PIL": latents_clean_PIL,
                        "zT": latents_PIL,
                        "message_bits_str_list": message_bits_str_list
                        }
        return results_dict


    def __decode(self, reversed_noise, K, target_payload_length, detection=False):
        """
            T2SMark decoding function that obtains the message bits from the watermarked noise
            
        """
        n = reduce(lambda x, y: x * y, reversed_noise.shape, 1)
        m = target_payload_length
        r = int(2 * norm.cdf(-self.tau) * n / m)
        k = m * r
        self.prng.manual_seed(self.__binlist2int(K))
        
        v_value = torch.randint(0, 2, (k,), generator=self.prng, device=self.device).float() * 2 - 1
        v_support = torch.randperm(n, generator=self.prng, device=self.device)[:k]

        w = torch.zeros(n, dtype=torch.bool, device=self.device)
        w[v_support] = True

        watermarked_vec = reversed_noise.flatten()[w] * v_value
        p = watermarked_vec.reshape(r, m).sum(dim=0)
        b = (p < 0).int()
        
        if detection:
            return b, torch.norm(p.flatten(), p=1).item()
        return b

    def __recover_messages_from_latents(self,
                                        latents: typing.Union[torch.Tensor, np.ndarray],
                                        master_key) -> dict:
        """
        Recover messages from latents

        @param latents: latent either tensor with batch dim
        @return: dict
        """
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        latents = latents.to(self.device)
        batch_size = latents.shape[0]
        recovered_messages_bits_str = []
        l1_norms = [] # Collect norms for the whole batch

        for i in range(batch_size):

            # Calculate the actual index in the latents tensor (0 to batch_size - 1)
            reversed_key_channel = latents[i, self.key_channel_idx, :, :]
            reversed_msg_channel = latents[i, self.msg_channel_idx, :, :]
        
            reversed_key, norm1_w = self.__decode(
                reversed_noise=reversed_key_channel, 
                K=master_key, 
                target_payload_length=self.key_length, 
                detection=True
            )
            
            reversed_msg = self.__decode(
                reversed_noise=reversed_msg_channel, 
                K=reversed_key, 
                target_payload_length=self.message_width_in_bits, 
                detection=False
            )

            # convert bits to string (cast to int first to avoid decimals like '1.0')
            message_bits_str = ''.join(str(int(bit.item())) for bit in reversed_msg)
            
            recovered_messages_bits_str.append(message_bits_str)
            l1_norms.append(norm1_w)
            
        return {
            "messages_bits_str": recovered_messages_bits_str,
            "l1_norm_w": l1_norms # Returns the list of norms for the entire batch
        }

    def __calculate_bit_accuracy(self,
                                 original_message_hex: typing.Any,
                                 extracted_message_bits_str: str) -> float:
        """
        Gett bit accuracy between extracted bits and the original message hex

        @param original_message_hex: original message in hex
        @param extracted_message_bits_str: extracted message in bits

        @return: bit accuracy
        """
        # Convert the original hex message to binary
        original_message_bin = ''.join(format(byte, '08b') for byte in original_message_hex)

        # Ensure both binary strings are of the same length
        min_length = min(len(original_message_bin), len(extracted_message_bits_str))
        original_message_bin = original_message_bin[:min_length]
        extracted_message_bits_str = extracted_message_bits_str[:min_length]
        
        # Calculate bit accuracy
        matching_bits = sum(1 for x, y in zip(original_message_bin, extracted_message_bits_str) if x == y)
        bit_accuracy = matching_bits / min_length
        
        return bit_accuracy


    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.array]) -> typing.Dict[str, typing.Any]:
        """
        Get bit accuracy between original and extracted messages

        @param latents: latent either tensor with batch dim or numpy with batch dim
        @return: dict
        """
        # get the extracted message
        recovered = self.__recover_messages_from_latents(latents, self.master_key)
        recovered_messages_bits_str = recovered["messages_bits_str"]
        l1_norm_w = recovered["l1_norm_w"]
        # iterate and calulate bit accuracies
        bit_accuracies = []
        recovered_message_bits_str_list = []
        batch_size = latents.shape[0]
        for i in range(batch_size):
            original_message_hex = self.message
            recovered_message_bits_str = recovered_messages_bits_str[i]

            # get the bit accuracy
            bit_accuracy = self.__calculate_bit_accuracy(original_message_hex, recovered_message_bits_str)
            bit_accuracies.append(bit_accuracy)
            recovered_message_bits_str_list.append(recovered_message_bits_str)

        return {
            "accuracies": bit_accuracies,
            "bit_accuracies": bit_accuracies,
            "message_bits_str_list": recovered_message_bits_str_list,
            "l1_norm_w": l1_norm_w
        }