# Adapted from https://github.com/XuandongZhao/PRC-Watermark

import typing
import argparse
import numpy as np
from scipy.special import erf
import torch
import pickle
from .sem_watermark import SemanticWatermark
import os
from scipy.sparse import csr_matrix
from scipy.special import binom
from pipe.image_utils import torch_to_PIL
from ldpc import bp_decoder
import galois
import sys
from pathlib import Path

DEFAULT_PRC_KEY_PATH = Path(__file__).resolve().parent / "prc_key.pkl"

GF = galois.GF(2)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--message_width_in_bytes', default=32, type=int, help="Number of bytes in the watermark payload")
parser.add_argument('--fpr', default=1e-6, type=float, help="Target False Positive Rate for PRC")
parser.add_argument('--t', default=3, type=int, help="Parity check weight (LDPC parameter)")
parser.add_argument('--prc_key_path', default=str(DEFAULT_PRC_KEY_PATH), type=str, help="Path to save or load the PRC keys (.pkl file)")

class PRCWatermark(SemanticWatermark):
    _shared_encoding_key = None
    _shared_decoding_key = None
    @classmethod
    def KeyGen(cls, n, message_length=512, false_positive_rate=1e-9, t=3, g=None, r=None, noise_rate=None):
        """
        Generate encoding and decoding keys for the PRC scheme.

        Args:
        n: Block length, i.e., the number of bits in a PRC codeword.
        message_length: Number of bits in the message to encode.
        false_positive_rate: Target false-positive rate.
        t: Number of nonzero entries per parity check. Larger values
            help pseudorandomness.
        g: Dimension of the random code. Larger values help
            pseudorandomness. If None, uses the computed security parameter.
        r: Number of parity checks. Smaller values help pseudorandomness.
            If None, computed from the other scheme parameters.
        noise_rate: Amount of noise added to codewords during encoding.
            Larger values help pseudorandomness. If None, computed from
            the security parameter and g.

        Returns:
        A tuple (encoding_key, decoding_key)
        """
        # Set basic scheme parameters
        num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
        secpar = int(np.log2(binom(n, t)))
        if g is None: g = secpar
        # if noise_rate is None: noise_rate = np.exp(lambertw(-np.log(2) / secpar, -1)).real
        # if noise_rate is None: noise_rate = 1 - 2**(-(secpar - 3*np.log2(g))/g**2)
        if noise_rate is None: noise_rate = 1 - 2 ** (-secpar / g ** 2)
        k = message_length + g + num_test_bits
        if r is None: r = n - k - secpar

        # Sample n by k generator matrix (all but the first n-r of these will be over-written)
        generator_matrix = GF.Random((n, k))

        # Sample scipy.sparse parity-check matrix together with the last n-r rows of the generator matrix
        row_indices = []
        col_indices = []
        data = []
        for row in range(r):
            chosen_indices = np.random.choice(n - r + row, t - 1, replace=False)
            chosen_indices = np.append(chosen_indices, n - r + row)
            row_indices.extend([row] * t)
            col_indices.extend(chosen_indices)
            data.extend([1] * t)
            generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
        parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

        # Compute scheme parameters
        max_bp_iter = int(np.log(n) / np.log(t))

        # Sample one-time pad and test bits
        one_time_pad = GF.Random(n)
        test_bits = GF.Random(num_test_bits)

        # Permute bits
        permutation = np.random.permutation(n)
        generator_matrix = generator_matrix[permutation]
        one_time_pad = one_time_pad[permutation]
        parity_check_matrix = parity_check_matrix[:, permutation]

        encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
        decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)

        return encoding_key, decoding_key

    def __init__(self, 
                 message_width_in_bytes: int = 32, 
                 message: typing.Optional[str] = None,
                 fpr: float = 1e-6, 
                 t: int = 3, 
                 offset: int = 0,
                 prc_key_path: str = None,
                 **kwargs):
        """
            Configure a PRC watermarking scheme. If a path is provided for the PRC keys, 
            it will attempt to load them from that path. 
            If the path does not exist, it will generate new keys and save them to that path.
            Args:
                message_width_in_bytes: Message length in bytes.
                offset: Index into the predefined message list
                message: Optional message overriding the predefined selection.
                prc_key_path: Optional path to the PRC keys.
                fpr: Target false positive rate for the PRC scheme.
                t: Parity check weight (LDPC parameter).
        """
        
        super().__init__(**kwargs)
        self.message_width_in_bytes = message_width_in_bytes
        self.message_width_in_bits = message_width_in_bytes * 8
        self.n = self.num_channels * self.latent_resolution * self.latent_resolution # block length
        self.t = t  # parameter which may be increased for improved undetectability at the cost of robustness.
        self.fpr = fpr
        self.offset = offset
        self.prc_key_path = prc_key_path
        if PRCWatermark._shared_encoding_key is None:
            if self.prc_key_path is not None and os.path.exists(self.prc_key_path):
                with open(self.prc_key_path, 'rb') as f:
                    keys = pickle.load(f)
                    encoding_key = keys['encoding_key']
                    decoding_key = keys['decoding_key']
            else:
                encoding_key, decoding_key = PRCWatermark.KeyGen(
                    n=self.n, 
                    message_length=self.message_width_in_bits, 
                    false_positive_rate=self.fpr, 
                    t=self.t,
                )
                # If a path was provided but didn't exist, save the newly generated keys
                if self.prc_key_path is not None:
                    print(f"Saving newly generated PRC keys to {self.prc_key_path}...")
                    # Ensure the directory exists before saving
                    os.makedirs(os.path.dirname(self.prc_key_path), exist_ok=True)
                    with open(self.prc_key_path, 'wb') as f:
                        pickle.dump({'encoding_key': encoding_key, 'decoding_key': decoding_key}, f)

            PRCWatermark._shared_encoding_key = encoding_key
            PRCWatermark._shared_decoding_key = decoding_key

        # Assign the shared keys to this specific instance
        self.encoding_key = PRCWatermark._shared_encoding_key
        self.decoding_key = PRCWatermark._shared_decoding_key

        # If message not given, use from the list, indexing using offset
        if message is None:
            from .messages import MESSAGES
            MESSAGES = [m[:self.message_width_in_bytes] for m in MESSAGES]  # trim to message_width_in_bytes to fit our required message length
            self.message = MESSAGES[offset]
        else:
            self.message = message


    def get_wm_type(self) -> str:
        return "PRC"


    def get_wm_latents(self, **kwargs) -> typing.Dict[str, typing.Any]:
        """
        Provide watermark latents and barcodes for the given batch size.
        All of the latents in the batch will carry the same watermark signal (for PRC it's same key and message).
        Generate randomized PRC latents sharing one message and encoding key.
        Random payload bits, encoding noise, and magnitudes vary per latent.
                        
        @return: dict
        """
        batch_size = self.batch_size # The number of images that would carry this watermark signal

        # Get message
        message_bytes = self.message 

        # Construct messages
        message_bits_str = ''.join(format(byte, '08b') for byte in message_bytes) 
        message_bits_str_list = [message_bits_str] * batch_size 

        # Get the message in bits
        message_bits = np.array([int(bit) for bit in message_bits_str])

        # Get the shared encoding key.
        generator_matrix, one_time_pad, test_bits, g, noise_rate = self.encoding_key
        n, k = generator_matrix.shape

        padding_length = k - len(test_bits) - g - len(message_bits)
        if padding_length < 0:
            raise ValueError("Message is too long")

        latents_torch = []
        for _ in range(batch_size):
            random_g = GF.Random(g)
            payload = np.concatenate((test_bits, random_g, GF(message_bits), GF.Zeros(padding_length)))
            error = GF(np.random.binomial(1, noise_rate, n))
    
            codeword = 1 - 2 * torch.tensor(payload @ generator_matrix.T + one_time_pad + error, dtype=torch.float32)
            codeword_np = codeword.numpy()

            latent_np = codeword_np * np.abs(np.random.randn(*codeword_np.shape))

            # Reshape to latent dimension (4 x 64 x 64)
            latent_torch = torch.from_numpy(latent_np).to(dtype=self.dtype, device=self.device).reshape(self.latent_shape[1:])
            latents_torch.append(latent_torch)

        # finalize
        latents_torch = torch.stack(latents_torch, dim=0)
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

    ### Given a GF(2) matrix, do row elimination and return the first k rows of A that form an invertible matrix
    def __boolean_row_reduce(self, A, print_progress=False):
        """
            Helper function for decoding in PRC
        """
        n, k = A.shape
        A_rr = A.copy()
        perm = np.arange(n)
        for j in range(k):
            idxs = j + np.nonzero(A_rr[j:, j])[0]
            if idxs.size == 0:
                print("The given matrix is not invertible")
                return None
            A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]] 
            (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])
            A_rr[idxs[1:]] += A_rr[j]
            if print_progress and (j%5==0 or j+1==k):
                sys.stdout.write(f'\rDecoding progress: {j + 1} / {k}')
                sys.stdout.flush()
        if print_progress: print()
        return perm[:k]

    def __recover_posteriors_from_latents(self, latents: torch.Tensor, basis=None, variances=None) -> torch.Tensor:
        # Use the same method as in prc_gaussians.recover_posteriors
        if variances is None:
            default_variance = 1.5
            denominators = np.sqrt(2 * default_variance * (1+default_variance)) * torch.ones_like(latents)
        elif type(variances) is float:
            denominators = np.sqrt(2 * variances * (1 + variances))
        else:
            denominators = torch.sqrt(2 * variances * (1 + variances))

        if basis is None:
            return erf(latents / denominators)
        else:
            return erf((latents @ basis) / denominators)

    def __detect_and_decode(self, latents: torch.Tensor, print_progress: bool = False) -> typing.Dict[str, typing.Any]:
        """
            Perform detection and decoding on a batch of latents. Returns a dictionary containing detection results and extracted messages.
            @param latents: A batch of latents to process.
            @param print_progress: Whether to print progress during processing.
            @return: A dictionary with keys "PRC_detection_results" and "messages_bits_str", 
            containing detection results and extracted messages respectively.
        """
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter, t = self.decoding_key
        fpr = false_positive_rate_key
        
        detection_results = []
        extracted_bits_strs = []

        # Iterate through the latents
        for i, latent in enumerate(latents):
            single_latent = latent.flatten().cpu()
            
            posteriors = self.__recover_posteriors_from_latents(single_latent)
            posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
            r = parity_check_matrix.shape[0]
            Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
            log_plus = np.log((1 + Pi) / 2)
            log_minus = np.log((1 - Pi) / 2)
            log_prod = log_plus + log_minus

            const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
            threshold = np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum()

            # 1. Store detection result for this image
            detection_result = log_plus.sum() >= threshold
            
            # Decode
            channel_probs = (1 - np.abs(posteriors)) / 2
            x_recovered = (1 - np.sign(posteriors)) // 2

            bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
            x_decoded = bpd.decode(x_recovered)

            # Compute a confidence score.
            bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
            confidences = 2 * np.abs(0.5 - bpd_probs)

            # Order codeword bits by confidence.
            confidence_order = np.argsort(-confidences)
            ordered_generator_matrix = generator_matrix[confidence_order]
            ordered_x_decoded = x_decoded[confidence_order]

            # Find the first linearly independent set of rows
            top_invertible_rows = self.__boolean_row_reduce(ordered_generator_matrix, print_progress=print_progress)
            
            # 2. Check for decoding failures safely
            if top_invertible_rows is None:
                detection_results.append(bool(detection_result))
                extracted_bits_strs.append(None)
                continue # Skip the rest of the math for this image

            # Solve the system.
            if print_progress:
                print(f"Solving linear system for batch item {i}...")
            recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows], GF(ordered_x_decoded[top_invertible_rows].astype(int)))
            decoding_success = bool((recovered_string[:len(test_bits)] == test_bits).all())
            

            combined_result = bool(detection_result or decoding_success)
            payload_array = np.array(recovered_string[len(test_bits) + g:])
            extracted_bits_str = ''.join(str(int(bit)) for bit in payload_array)
            
            detection_results.append(combined_result)
            extracted_bits_strs.append(extracted_bits_str)

        # Return the lists packaged nicely in a dict
        return {
            "PRC_detection_results": detection_results,
            "messages_bits_str": extracted_bits_strs
        }

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

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.array]) -> typing.Dict[str, any]:
        """
        Get bit accuracy between original and extracted messages
        """
        # Get the extracted messages for the whole batch
        batch_size = latents.shape[0]
        recovered = self.__detect_and_decode(latents)
        detection_results = recovered["PRC_detection_results"]
        recovered_messages_bits_str = recovered["messages_bits_str"]

        bit_accuracies = []
        recovered_message_bits_str_list = []
        
        # Iterate and calculate bit accuracies
        for i in range(batch_size):
            original_message_hex = self.message
            extracted_bits_str = recovered_messages_bits_str[i]

            # Handle the case where decoding failed (returned None)
            if extracted_bits_str is None:
                bit_accuracies.append(0.0)
                recovered_message_bits_str_list.append("")
            else:
                bit_accuracy = self.__calculate_bit_accuracy(
                    original_message_hex=original_message_hex, 
                    extracted_message_bits_str=extracted_bits_str
                )
                bit_accuracies.append(bit_accuracy)
                recovered_message_bits_str_list.append(extracted_bits_str)

        return {
            "accuracies": bit_accuracies,
            "PRC_detection_results": detection_results,
            "bit_accuracies": bit_accuracies,
            "message_bits_str_list": recovered_message_bits_str_list
        }


    