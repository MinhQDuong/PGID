import torch
import numpy as np
import random
from watermarks.watermark_utils import check_if_detection_successful

def set_random_seed(seed=0):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def configure_prc(args, pipe_provider_target):
    """Apply PRC-specific settings to the target pipeline.
    Based on https://github.com/XuandongZhao/PRC-Watermark
    """
    if args.wm_type != "PRC":
        return

    assert args.scheduler_target == "DPM", (
        "PRC requires --scheduler_target DPM"
    )

    args.guidance_scale_target = 3.0

    base_config = pipe_provider_target.scheduler.config
    forward_class = type(pipe_provider_target.scheduler)
    inverse_class = type(pipe_provider_target.scheduler_inverse)

    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "steps_offset": 1,
        "solver_order": 1,
    }

    pipe_provider_target.scheduler = forward_class.from_config(
        base_config,
        **scheduler_kwargs,
    )
    pipe_provider_target.scheduler_inverse = inverse_class.from_config(
        base_config,
        **scheduler_kwargs,
    )

    # Synchronize the underlying pipeline if already loaded.
    if pipe_provider_target.pipe is not None:
        pipe_provider_target.pipe.scheduler = pipe_provider_target.scheduler


def configure_PGID(args):
    """Main experiment parameters for PGID"""
    args.num_defense_steps = 50

    if args.attack_type == "removal":
        args.stopping_timestep = 10
        args.num_skip_inversion = 1
        args.guidance_strength = 0.045

    elif args.attack_type == "forgery":
        args.stopping_timestep = 7 if args.wm_type == "PRC" else 15
        args.num_skip_inversion = 3
        args.guidance_strength = 0.001

    else:
        raise ValueError(
            f"Unsupported attack type: {args.attack_type}"
        )


def get_watermark_args(args, device):
    """Return the selected watermark's constructor arguments."""
    argument_names = {
        "GS": (
            "message_width_in_bytes",
            "num_replications",
            "l",
        ),
        "TR": (
            # w_seed is passed separately to generate_watermark_signals().
            "w_channel",
            "w_pattern",
            "w_mask_shape",
            "w_radius",
            "w_measurement",
            "w_injection",
            "w_pattern_const",
        ),
        "PRC": (
            "message_width_in_bytes",
            "fpr",
            "t",
            "prc_key_path",
        ),
        "T2SMark": (
            "message_width_in_bytes",
            "tau",
            "key_length",
            "key_channel_idx",
        ),
    }

    if args.wm_type not in argument_names:
        raise ValueError(f"Unsupported watermark type: {args.wm_type}")

    watermark_args = {
        name: getattr(args, name)
        for name in argument_names[args.wm_type]
    }
    watermark_args["device"] = device
    return watermark_args

def add_detection_result(results, wm_type, detection_threshold):
    """Return evaluation metrics with a unified detection boolean."""
    results = dict(results)

    if wm_type == "PRC":
        results["detection_successful"] = results["PRC_detection_results"]
    else:
        metric_names = {
            "GS": "bit_accuracy",
            "TR": "p_value",
            "T2SMark": "l1_norm_w",
        }

        if wm_type not in metric_names:
            raise ValueError(f"Unsupported watermark type: {wm_type}")

        value = results[metric_names[wm_type]]

        results["detection_successful"] = check_if_detection_successful(
            wm_type=wm_type,
            threshold=detection_threshold,
            value=value,
        )

    return results



def save_result(all_results, image_idx, stage, attack_step, metrics):
    """Add one evaluation result to the list of all results."""
    row = dict(metrics)
    row.update({
        "image_idx": image_idx,
        "stage": stage,
        "attack_step": attack_step,
    })

    all_results.append(row)
    return all_results

def summarize_results(results_df, wm_type):
    """Calculate mean metrics for each stage and attack checkpoint."""
    metric_names = {
        "GS": ["bit_accuracy"],
        "TR": ["p_value"],
        "PRC": [],
        "T2SMark": ["bit_accuracy", "l1_norm_w"],
    }

    if wm_type not in metric_names:
        raise ValueError(f"Unsupported watermark type: {wm_type}")

    aggregations = {
        "num_images": ("image_idx", "count"),
        "detection_rate": ("detection_successful", "mean"),
    }

    # Include scheme-specific metrics when present.
    for metric in metric_names[wm_type]:
        if metric in results_df.columns:
            aggregations[f"mean_{metric}"] = (metric, "mean")

    return (
        results_df
        .groupby(["stage", "attack_step"], sort=False, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

def print_metrics(label, results):
    message = f"{label}: detection={results['detection_successful']}"
    for metric in ("bit_accuracy", "p_value", "l1_norm_w"):
        value = results.get(metric)
        if value is not None:
            message += f", {metric}={value:.6g}"
    print(message)