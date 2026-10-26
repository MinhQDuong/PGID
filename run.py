# Main script to run attack and PGID defense
import os
import PIL
import torch
import pandas as pd
import argparse
from watermarks.watermark_utils import SemanticWatermarks, get_detection_threshold, validate
from watermarks.gs_watermark import parser as gs_parser
from watermarks.tr_watermark import parser as tr_parser
from watermarks.prc_watermark import parser as prc_parser
from watermarks.t2smark_watermark import parser as t2smark_parser
from pgid_defense import PGIDPipeline, parser as pgid_parser
from pipe import attack_pipe, pipe_utils
from utils import print_metrics, set_random_seed, configure_prc, configure_PGID, get_watermark_args, add_detection_result, save_result, summarize_results
from imprint_attacks import ImprintForgeryAttack, ImprintRemovalAttack
from prompts import PROMPTS_LIST

# Args
parser = argparse.ArgumentParser(description="Perform attack and defense", parents=[gs_parser, tr_parser, prc_parser, t2smark_parser, pgid_parser], conflict_handler='resolve') 
parser.add_argument("--out_dir", type=str, default="out_results", help="Directory to save results")
parser.add_argument("--attack_type", type=str, default="removal", choices=["removal", "forgery"])

# For forgery attacks
parser.add_argument("--cover_images_dir", type=str, default="cover_images/ms_coco", help="Directory with cover images for forgery")

# Target model
parser.add_argument("--modelid_target",
                    type=str,
                    default="Manojb/stable-diffusion-2-1-base",
                    choices=["Manojb/stable-diffusion-2-1-base", "PixArt-alpha/PixArt-XL-2-512x512"])
parser.add_argument("--scheduler_target", type=str, default="DDIM", choices=["DDIM", "DPM"])
parser.add_argument("--num_inference_steps_target", type=int, default=50)
parser.add_argument("--guidance_scale_target", type=float, default=7.5)

# Attacker model
parser.add_argument("--modelid_attacker", type=str, default="Manojb/stable-diffusion-2-1-base", choices=["stable-diffusion-v1-5/stable-diffusion-v1-5", "Manojb/stable-diffusion-2-1-base"])
parser.add_argument("--scheduler_attacker", type=str, default="DDIM", choices=["DDIM", "DPM"])
parser.add_argument("--num_inference_steps_attacker", type=int, default=50)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--steps", type=int, default=151)

# Others
parser.add_argument("--wm_type",
                    type=str,
                    default="GS",
                    choices=[wm.name for wm in SemanticWatermarks])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_images", type=int, default=1, help="Number of images to process")
parser.add_argument("--start_index", type=int, default=0) # index to start processing images from
parser.add_argument("--gpu", type=str, default="cuda", help="GPU device to use")

args = parser.parse_args()

DEVICE = args.gpu if torch.cuda.is_available() else "cpu"
TARGET_SAVE_STEPS = (50, 100, 150)


assert max(TARGET_SAVE_STEPS) <= args.steps, "Max target save step must be <= total steps"
set_random_seed(args.seed)
detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)


# Load pipeline which generates the watermark
pipe_provider_target = pipe_utils.create_diffusion_pipe(
    pretrained_model_name_or_path=args.modelid_target,
    resolution=args.resolution,
    device=DEVICE,
    scheduler_name=args.scheduler_target,
    scheduler_inverse_name=args.scheduler_target,
    eager_loading=True,
    disable_tqdm=True,
)

if args.wm_type == "PRC":
    configure_prc(args, pipe_provider_target)

# Load attack pipe
pipe_attacker, forward_scheduler, inverse_scheduler = (
    attack_pipe.create_attack_pipe(
        pretrained_model_name_or_path=args.modelid_attacker,
        resolution=args.resolution,
        device=DEVICE,
        scheduler_name=args.scheduler_attacker,
        scheduler_inverse_name=args.scheduler_attacker,
    )
)

# Process prompts
prompts = PROMPTS_LIST[args.start_index:args.start_index + args.num_images]

# Create output directories to save attacked images and watermarked images
out_dir = os.path.join(args.out_dir, f"attack={args.attack_type}")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "watermarked_images"), exist_ok=True)

# Create a separate directory for each target step dynamically
for step in TARGET_SAVE_STEPS:
    os.makedirs(os.path.join(out_dir, f"attacked_images_step_{step}"), exist_ok=True)

# Load cover images for forgery attack if attack_type is forgery
cover_images = []
if args.attack_type == "forgery":
    if args.cover_images_dir is None:
        raise ValueError("Must provide --cover_images_dir for forgery attack")
    cover_image_files = sorted([f for f in os.listdir(args.cover_images_dir) 
                                if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[args.start_index:args.start_index + args.num_images]
    for img_file in cover_image_files:
        img_path = os.path.join(args.cover_images_dir, img_file)
        cover_img = PIL.Image.open(img_path).resize((args.resolution, args.resolution)).convert('RGB')
        cover_images.append(cover_img)
    print(f"Loaded {len(cover_images)} cover images for forgery attack")

# Some checks to simplify running
if len(prompts) != args.num_images:
    raise ValueError("Not enough prompts for the requested images.")

if args.attack_type == "forgery" and len(cover_images) != args.num_images:
    raise ValueError("Not enough cover images for the requested images.")

watermark_args = get_watermark_args(args, DEVICE) # Get the watermark constructor arguments based on the watermark type
print(f"\nStarting {args.attack_type} attack processing for {len(prompts)} prompts (indices {args.start_index} to {args.start_index + len(prompts) - 1})...\n")

# Collect all results in this
all_results = [] 

# Prepare PGID defense pipeline
configure_PGID(args)

PGID_defense = PGIDPipeline(
    pipe=pipe_provider_target,
    scheduler_config=pipe_provider_target.scheduler.config,
    num_inference_steps=args.num_defense_steps,
    stopping_timestep=args.stopping_timestep,
    num_skip_inversion=args.num_skip_inversion,
    guidance_strength=args.guidance_strength,
)

# Generate a watermark signal for each image
for local_idx, (sem_watermark, idx) in enumerate(
    SemanticWatermarks[args.wm_type].value.generate_watermark_signals(
        total_num_signals=args.start_index + args.num_images,
        latent_shape=pipe_provider_target.get_latent_shape(),
        batch_size=1,
        target_start_index=args.start_index,
        target_end_index=args.start_index + args.num_images,
        w_seed=args.seed,
        **watermark_args,
    )
):
    prompt = prompts[local_idx]
    print(f"Processing image {idx+1}/{args.start_index + len(prompts)}")

    wm_initial_results = sem_watermark.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]
    message_bits_str_initial = wm_initial_results.get("message_bits_str_list", [None])[0]
    generated_PIL_list = pipe_provider_target.generate(
        prompts=prompt,
        latents=wm_zT,
        num_inference_steps=args.num_inference_steps_target,
        guidance_scale=args.guidance_scale_target
    )["images_PIL"]
    generated_PIL = generated_PIL_list[0]
    generated_pt = pipe_provider_target.PIL_to_torch(generated_PIL_list)
    print(f"Watermarked image generated.")

    # Save watermarked image
    watermarked_path = os.path.join(out_dir, "watermarked_images", f"image_{idx:03d}.png")
    generated_PIL.save(watermarked_path)

    results_wm = validate(
        image_to_verify_PIL=generated_PIL,
        watermark=sem_watermark,
        pipe_provider_target=pipe_provider_target,
        num_inference_steps_target=args.num_inference_steps_target,
        message_bits_str_initial=message_bits_str_initial,
    )

    results_wm = add_detection_result(results_wm, args.wm_type, detection_threshold)
    all_results = save_result(all_results, image_idx=idx, stage="watermarked", metrics=results_wm, attack_step=-1)
    print_metrics("Watermarked", results_wm)

    ### Perform attack
    # Removal modifies the watermarked image.
    # Forgery modifies the cover image.
    if args.attack_type == "removal":
        attack_class = ImprintRemovalAttack
        image_to_attack = generated_PIL
    else:
        attack_class = ImprintForgeryAttack
        image_to_attack = cover_images[local_idx]

    imprint_attack = attack_class(
        inverse_scheduler=inverse_scheduler,
        attack_pipe=pipe_attacker,
        image=image_to_attack,
        reference_image=generated_PIL,
        learning_rate=args.lr,
        num_inference_steps_attacker=args.num_inference_steps_attacker,
        steps=args.steps,
        save_steps=TARGET_SAVE_STEPS,
    )

    attack_results = imprint_attack.run()

    # Decode, save, and evaluate each attack checkpoint.
    for attack_step in TARGET_SAVE_STEPS:
        attacked_z0 = attack_results["saved_latents"][attack_step]

        attacked_PIL = attack_pipe.latent_to_pil(attacked_z0, pipe_attacker)[0]

        attacked_path = os.path.join(out_dir, f"attacked_images_step_{attack_step}", f"image_{idx:03d}.png")
        attacked_PIL.save(attacked_path)

        results_attack = validate(
            image_to_verify_PIL=attacked_PIL,
            watermark=sem_watermark,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            message_bits_str_initial=message_bits_str_initial,
        )

        results_attack = add_detection_result(
            results_attack,
            args.wm_type,
            detection_threshold,
        )

        all_results = save_result(
            all_results,
            image_idx=idx,
            stage="attacked",
            attack_step=attack_step,
            metrics=results_attack,
        )

        print_metrics(f"Attack Step {attack_step}", results_attack)

        defense_output = PGID_defense.run_defense(attacked_PIL)
        defended_zT = defense_output["zT"]

        # Evaluate PGID's returned latent directly.
        results_defense = validate(
            image_to_verify_PIL=attacked_PIL,
            watermark=sem_watermark,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            message_bits_str_initial=message_bits_str_initial,
            provided_zT=defended_zT, # Use defended_zT directly for watermark detection
        )

        results_defense = add_detection_result(
            results_defense,
            args.wm_type,
            detection_threshold,
        )

        all_results = save_result(
            all_results,
            image_idx=idx,
            stage="defended",
            attack_step=attack_step,
            metrics=results_defense,
        )
        print_metrics(f"PGID Defense on Attack Step {attack_step}", results_defense)

results_df = pd.DataFrame(all_results)
results_df = results_df.dropna(axis=1, how="all")

results_df.to_csv(
    os.path.join(out_dir, "all_results.csv"),
    index=False,
)

summary_df = summarize_results(results_df, args.wm_type)
summary_df.to_csv(
    os.path.join(out_dir, "summary.csv"),
    index=False,
)
print(summary_df.to_string(index=False))
