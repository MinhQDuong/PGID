# Originally from https://github.com/lthero-big/A-watermark-for-Diffusion-Models
# We modify code from https://github.com/and-mill/semantic-forgery - Use watermark signal approach instead of provider

import typing
import argparse
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
import numpy as np
from scipy.stats import norm
import torch
from .sem_watermark import SemanticWatermark
from pipe.image_utils import torch_to_PIL

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--l', default=1, type=int, help="The size of slide windows for m")
parser.add_argument('--num_replications', default=64, type=int, help="The number of replications of the message bits to get barcode image")
parser.add_argument('--message_width_in_bytes', default=32, type=int, help="Message width in bytes")


class GSWatermark(SemanticWatermark):
    def __init__(self,
                 message_width_in_bytes: int = 32,  # (channels/f_c) * f_h * f_w // 8
                 num_replications: int = 64,
                 l: int = 1,
                 offset: int = 0, # Used to choose the message, key and nonce from the list. Different offset will result in different watermark signals.
                 message: typing.Optional[str] = None,
                 key: str = None,
                 nonce: str = None,
                 **kwargs):
        """
            Configure a Gaussian Shading watermark signal.

            Uses predefined messages, keys, and nonces (from the list).

            The parameters must satisfy:
                message_width_in_bytes * 8 * num_replications / l
                == num_channels * latent_resolution**2

            Args:
                message_width_in_bytes: Message length in bytes.
                num_replications: Number of message repetitions for error correction.
                l: Window size controlling noise reconstruction tolerance.
                offset: Index into the predefined message, key, and nonce lists.
                message: Optional message overriding the predefined selection.
                key: Optional encryption key overriding the predefined selection.
                nonce: Optional nonce overriding the predefined selection.
        """
        super().__init__(**kwargs)

        self.message_width_in_bytes = message_width_in_bytes
        self.message_width_in_bits = int(message_width_in_bytes * 8)
        self.barcode_width = self.message_width_in_bits

        self.num_replications = num_replications
        self.l = l
        self.offset = offset  

        assert self.message_width_in_bits * self.num_replications // self.l == self.num_channels * self.latent_resolution * self.latent_resolution

        # If message not given, use from the list, indexing using offset
        if message is None:
            from .messages import MESSAGES
            MESSAGES = [m[:self.message_width_in_bytes] for m in MESSAGES]  # trim to message_width_in_bytes to fit our required message length
            from .keys import KEYS
            from .nonces import NONCES
            self.message = MESSAGES[offset]
            self.key = KEYS[offset]
            self.nonce = NONCES[offset]

        # Custom needs all three of message, key and nonce to be provided
        else:
            assert key is not None and nonce is not None
            self.message = message
            self.key = key
            self.nonce = nonce


    def get_wm_type(self) -> str:
        return "GS"


    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        """
        Provide watermark latents and barcodes for the given batch size.
        All of the latents in the batch will carry the same watermark signal.
        Each latent is sampled differently but based on the same encrypted message, key and nonce.
                        
        @return: dict
        """
        batch_size = self.batch_size # The number of images that would carry this watermark signal

        # get the message, key and nonce
        message_bytes = self.message  # message_bytes = k
        key_bytes = self.key
        nonce_bytes = self.nonce

        # Construct messages
        message_bits_str = ''.join(format(byte, '08b') for byte in message_bytes) 
        message_bits_str_list = [message_bits_str] * batch_size 


        # Replicate the message bits to create a barcode
        replicated_message_bytes = message_bytes * self.num_replications
        replicated_message_bits_str = ''.join(
            format(byte, '08b') for byte in replicated_message_bytes
        )

        # Make barcode as np from s_d
        barcode_ints_2d_np = np.array([int(b, 2) for b in replicated_message_bits_str]).reshape(self.num_replications, self.message_width_in_bits)
        barcode_ints_2d_torch = torch.tensor(barcode_ints_2d_np, dtype=torch.uint8, device=self.device)
        barcodes_torch = barcode_ints_2d_torch.unsqueeze(0).expand(batch_size, -1, -1)  # Expand to batch size

        # Encrypt barcode using ChaCha20
        cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce_bytes), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_bytes = encryptor.update(replicated_message_bytes) + encryptor.finalize()  # encrypted_bytes = m
        encrypted_bits_str = ''.join(format(byte, '08b') for byte in encrypted_bytes)  # encrypted_bits_str = m_bits

        # Embed the encrypted message into the latent space using Gaussian sampling
        # Each latent zT is sampled differently but based on the same encrypted message, key and nonce. 
        # This is because we want to have different latents for each image in the batch, but they all carry the same watermark signal.
        latents_torch = []
        for _ in range(batch_size):
            pixel_values = []
            for i in range(0, len(encrypted_bits_str), self.l):
                window = encrypted_bits_str[i:i + self.l]
                y = int(window, 2)  # Convert the binary sequence (base 2) inside the window into integer y
                u = np.random.uniform(0, 1)  # Generate random u
                # Calculate z^s_T
                pixel_value = norm.ppf((u + y) / 2**self.l)
                pixel_values.append(pixel_value)

            pixel_values = torch.tensor(pixel_values, dtype=self.dtype, device=self.device)
            latent_torch = pixel_values.reshape(self.latent_shape[1:])  # latent shape without batch dim
            latents_torch.append(latent_torch)
        
        # finalize
        latents_torch = torch.stack(latents_torch, dim=0)

        latents_PIL = torch_to_PIL(latents_torch)
        barcodes_PIL = torch_to_PIL(barcodes_torch)
        
        # Clean latent for comparing maybe
        latents_clean = torch.randn_like(latents_torch, dtype=self.dtype, device=self.device)
        latents_clean_PIL = torch_to_PIL(latents_clean)

        results_dict = {"zT_torch": latents_torch,
                        "zT_clean_torch": latents_clean,
                        "zT_PIL": latents_PIL,
                        "zT_clean_PIL": latents_clean_PIL,
                        "zT": latents_PIL,
                        "barcodes_torch": barcodes_torch,
                        "barcodes_PIL": barcodes_PIL,
                        "barcodes": barcodes_PIL,
                        "message_bits_str_list": message_bits_str_list
                        }
    
        return results_dict
    

    def __recover_messages_from_latents(self,
                                        latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
        """
        Recover messages from zT latents

        @param latents: zT latent tensor with batch dim
        @return: dict
        """
        latents = latents.detach().cpu().numpy()
        key_bytes = self.key
        nonce_bytes = self.nonce

        recovered_messages_bits_str = []
        recovered_barcodes_torch = []

        for latent in latents:
            # Extract the encrypted message from the latent space via reverse Gaussian Sampling
            encrypted_numbers = []
            for zsT_pixel_value in np.nditer(latent):
                # Use the inverse operation of norm.ppf to recover the original y value
                y_reconstructed = norm.cdf(zsT_pixel_value) * 2**self.l
                y_reconstructed = int(y_reconstructed)
                # in some very rare and sporadic cases, y_reconstructed can be a 2 for l=1.
                if y_reconstructed == 2**self.l:
                    y_reconstructed -= 1
                encrypted_numbers.append(y_reconstructed)

            # For l > 1 we need another step that unfold number up to 2**l -1 to binary strings
            encrypted_bits = []
            for n in encrypted_numbers:
                # Convert the integer to binary and pad with zeros
                encrypted_bits.append(bin(n)[2:].zfill(self.l))
            encrypted_bits_str = ''.join(encrypted_bits)  # encrypted_bits_str = reconstructed_m_bits

            # Convert binary bits to bytes
            encrypted_bytes = bytes(
            int(
                    ''.join(
                        str(bit) for bit in encrypted_bits_str[j:j+8]
                        ),
                        2
                    ) for j in range(0, len(encrypted_bits_str), 8)
            )

            # Decrypt the message using ChaCha20 with the same key and nonce
            cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce_bytes), mode=None, backend=default_backend())
            decryptor = cipher.decryptor()
            replicated_message_bytes = decryptor.update(encrypted_bytes) + decryptor.finalize()  # replicated_message_bytes = s_d_reconstructed
            
            # Convert the decrypted byte string into a binary representation
            replicated_message_bits_str = ['{:08b}'.format(byte) for byte in replicated_message_bytes]  # replicated_message_bits_str = bits_list
            # Merge the binary strings into a long string
            all_bits = ''.join(replicated_message_bits_str)
            # list of strings with len self.message_width_in_bits
            barcode_string_1d = [all_bits[i:i+self.message_width_in_bits] for i in range(0,
                                                                                         len(all_bits),
                                                                                         self.message_width_in_bits)]  # barcode_string_2d = segments
            # make torch
            # 1st dim: number of elements in barcode_string_1d
            # 2nd dim: number of character ("0" or "1") in each element, these are self.message_width_in_bits
            barcode_int_2d_torch = torch.tensor([list(map(int, list(row))) for row in barcode_string_1d], dtype=torch.uint8, device=self.device)
    
            # Recover the original message bits by majority voting across the replications
            message_bits_str = ''
            for i in range(self.message_width_in_bits):
                # Calculate the count of '1's for each bit across all lines
                count_1 = sum(column[i] == '1' for column in barcode_string_1d)
                # for l>1, we might get sporadic errors becasue of numerical
                message_bits_str += '1' if count_1 > len(barcode_string_1d) / 2 else '0'

            # append
            recovered_messages_bits_str.append(message_bits_str)
            recovered_barcodes_torch.append(barcode_int_2d_torch)

        # finalize
        recovered_barcodes_torch = torch.stack(recovered_barcodes_torch, dim=0)
        recovered_barcodes_PIL = torch_to_PIL(recovered_barcodes_torch)

        return {"messages_bits_str": recovered_messages_bits_str,
                "barcodes_torch": recovered_barcodes_torch,
                "barcodes_PIL": recovered_barcodes_PIL}
    

    def __calculate_bit_accuracy(self,
                                 original_message_hex: typing.Any,
                                 extracted_message_bits_str: str) -> float:
        """
        Get bit accuracy between extracted bits and the original message hex

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
    

    def get_accuracies(self, latents: typing.Union[torch.Tensor]) -> typing.Dict[str, any]:
        """
        Get bit accuracy between original and extracted messages

        @param latents: latent tensor with batch dim
        @return: dict
        """
        # get the extracted message
        recovered = self.__recover_messages_from_latents(latents)
        recovered_messages_bits_str = recovered["messages_bits_str"]
        recovered_barcodes_torch = recovered["barcodes_torch"]
        recovered_barcodes_PIL = recovered["barcodes_PIL"]

        # iterate and calulate bit accuracies
        batch_size = latents.shape[0]
        bit_accuracies = []
        recovered_message_bits_str_list = []
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
            "barcodes_torch": recovered_barcodes_torch,
            "barcodes_PIL": recovered_barcodes_PIL,
            "barcodes": recovered_barcodes_PIL,
            "message_bits_str_list": recovered_message_bits_str_list
        }