# Adapted from https://github.com/and-mill/semantic-forgery

from enum import Enum
import torch
import PIL.Image
from watermarks.sem_watermark import SemanticWatermark
from watermarks.gs_watermark import GSWatermark
from watermarks.tr_watermark import TRWatermark
from watermarks.prc_watermark import PRCWatermark
from watermarks.t2smark_watermark import T2SMarkWatermark


class SemanticWatermarks(Enum): # Class to easily choose the watermark type
    GS = GSWatermark
    TR = TRWatermark
    PRC = PRCWatermark
    T2SMark = T2SMarkWatermark

# Gaussian Shading FPR=10-6 
# This is the threshold for the multi-bit setting with 100k users - from https://github.com/and-mill/semantic-forgery
GS_THRESHOLDS = 0.70703125
# Tree-Ring FPR=1%
# empricially measured with 5000 watermarked vs 5000 non-watermarked images on each model
TR_THRESHOLD_SD_2_1 = 0.053849971492007605
TR_THRESHOLD_PIXART_ALPHA = 0.04742941214204914
TR_THRESHOLD_FOR_MODEL = {
    "Manojb/stable-diffusion-2-1-base": TR_THRESHOLD_SD_2_1,
    "PixArt-alpha/PixArt-XL-2-512x512": TR_THRESHOLD_PIXART_ALPHA
}
# T2SMark FPR=10-6
# Calculated from 10000 watermarked vs 10000 non-watermarked images on each model and gaussian assumption
TSMARK_THRESHOLD_SD_2_1 = 260.7018
TSMARK_THRESHOLD_PIXART_ALPHA = 250.6571
TSMARK_THRESHOLD_FOR_MODEL = {
    "Manojb/stable-diffusion-2-1-base": TSMARK_THRESHOLD_SD_2_1,
    "PixArt-alpha/PixArt-XL-2-512x512": TSMARK_THRESHOLD_PIXART_ALPHA
}

def get_detection_threshold(wm_type: str,
                            model_name: str = None) -> float:
    """
    Get the detection threshold for the given model

    @param wm_type: str
    @param model_name: str

    @return: float
    """
    if wm_type == "GS":
        return GS_THRESHOLDS
    elif wm_type == "TR":
        assert model_name, "Model name must be provided for TR"
        return TR_THRESHOLD_FOR_MODEL.get(model_name, TR_THRESHOLD_SD_2_1)
    elif wm_type == "PRC":
        # We do not have a fixed threshold for PRC, as it depends on the false positive rate and other parameters. 
        # The detection function will calculate the threshold based on the provided false positive rate.
        return None
    elif wm_type == "T2SMark":
        assert model_name, "Model name must be provided for T2SMark"
        return TSMARK_THRESHOLD_FOR_MODEL.get(model_name, TSMARK_THRESHOLD_SD_2_1)
    else:
        raise ValueError("Unknown watermark type")
    

def check_if_detection_successful(wm_type: str, threshold: float, value: float) -> bool:
    """
        Check if the detection is successful

        @param wm_type: str
        @param threshold: float
        @param value: float

        @return: bool
    """
    if wm_type == "GS":
        return value > threshold
    elif wm_type == "TR":
        return value < threshold
    elif wm_type == "PRC":
        return value
    elif wm_type == "T2SMark":
        return value > threshold
    else:
        raise ValueError("Unknown watermark type")


def validate(
        image_to_verify_PIL: PIL.Image, # The image to verify whether it has watermark
        watermark: SemanticWatermark,
        pipe_provider_target,
        num_inference_steps_target,
        message_bits_str_initial = None,
        provided_zT = None, # Directly pass the noise latents, using this if provided
        ):
        """
        Perform a validation with the target model provider and the watermark.
        If provided_zT is not None, it will be used instead of inverting the image to get zT.
        Work on only one image at a time.

        @param image_to_verify_PIL: PIL.Image, the image to verify
        @param watermark: SemanticWatermark, the watermark
        @param pipe_provider_target: DiffusionPipelineProvider, the target model provider
        @param num_inference_steps_target: int, the number of inference steps for the target model
        @param message_bits_str_initial: str, the initial message bits
        @param provided_zT: torch.Tensor, the provided zT - overrides the inversion of the image if provided

        @return: dict, the results of the validation
        """

        if provided_zT is not None:
            zT_retrieved = provided_zT
        else:
            # If is an instance of PRCWatermark, we use exact inversion
            if isinstance(watermark, PRCWatermark):
                zT_retrieved = pipe_provider_target.exact_inversion(
                    images=image_to_verify_PIL,
                    num_inference_steps=num_inference_steps_target,
                    decode_inv=True, 
                )["zT_torch"]
                zT_retrieved = zT_retrieved.detach()
            else:
                with torch.no_grad():
                    # retrieve zT
                    zT_retrieved = pipe_provider_target.invert_images(image_to_verify_PIL, num_inference_steps=num_inference_steps_target)["zT_torch"]
        
        # watermark test
        accuracy_results = watermark.get_accuracies(zT_retrieved)

        # Gaussian Shading bit accuracy
        bit_accuracy = accuracy_results["bit_accuracies"][0] if "bit_accuracies" in accuracy_results else None
        message_bits_str_recovered = accuracy_results["message_bits_str_list"][0] if "message_bits_str_list" in accuracy_results else None

        # Tree-Ring p-value
        p_value = accuracy_results["p_values"][0] if "p_values" in accuracy_results else None

        # PRC detection result
        prc_detection_results = accuracy_results["PRC_detection_results"][0] if "PRC_detection_results" in accuracy_results else None

        # T2SMark l1-norm 
        l1_norm = accuracy_results["l1_norm_w"][0] if "l1_norm_w" in accuracy_results else None

        # collect metrics to a row for a csv with results
        return {
            "message_bits_str_initial": message_bits_str_initial,
            "bit_accuracy": bit_accuracy,
            "p_value": p_value, # For Tree Ring 
            "message_bits_str_recovered": message_bits_str_recovered,
            "PRC_detection_results": prc_detection_results, # For PRC
            "l1_norm_w": l1_norm # For T2SMark
        }