# PGID Defense Pipeline
import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler
import argparse

parser = argparse.ArgumentParser(description="PGID Defense")
parser.add_argument("--num_defense_steps", type=int, default=50, help="Number of DDIM inversion steps for the defense")
parser.add_argument("--stopping_timestep", type=int, default=10, help="Stopping timestep for progressive refinement (k)")
parser.add_argument("--num_skip_inversion", type=int, default=1, help="Number of inversion steps to skip during progressive refinement (s)")
parser.add_argument("--guidance_strength", type=float, default=0.045, help="Strength of the guidance signal")


class PGIDPipeline:
    def __init__(
        self,
        pipe, # A SDPipe or PixArtPipe
        scheduler_config,
        num_inference_steps: int = 50,
        stopping_timestep: int = 10,
        num_skip_inversion: int = 1,
        guidance_strength: float = 0.045,
        timestep_spacing=None,
    ):
        self.pipe = pipe

        self.num_inference_steps = num_inference_steps
        self.stopping_timestep = stopping_timestep
        self.num_skip_inversion = num_skip_inversion
        self.guidance_strength = guidance_strength

        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")

        if not 0 <= stopping_timestep < num_inference_steps:
            raise ValueError(
                "stopping_timestep must be non-negative and "
                "strictly less than num_inference_steps." # keep k < num_inference_steps
            )

        if num_skip_inversion < 0:
            raise ValueError("num_skip_inversion must be at least 0.")

        scheduler_kwargs = {}
        if timestep_spacing is not None:
            scheduler_kwargs["timestep_spacing"] = timestep_spacing

        self.inverse_scheduler = DDIMInverseScheduler.from_config(
            scheduler_config,
            **scheduler_kwargs,
        )
        self.forward_scheduler = DDIMScheduler.from_config(
            scheduler_config,
            **scheduler_kwargs,
        )

    def _step(self, z, timestep, scheduler, conditioning):
        """Perform one inversion or denoising update."""
        latent_input = scheduler.scale_model_input(z, timestep)

        noise_pred = self.pipe.predict_noise_with_null(
            latent_input,
            timestep,
            conditioning,
        )

        return scheduler.step(
            noise_pred,
            timestep,
            z,
            return_dict=False,
        )[0]

    # Used for stages I and III.
    def _invert(self, z0, conditioning, store_trace=False):
        """Fully invert z0, optionally recording the attacked trajectory."""
        z = z0.clone()
        trace = {}

        for timestep in self.inverse_scheduler.timesteps:
            if store_trace:
                # Store attacked trajectory
                trace[timestep.item()] = z.detach().clone()

            z = self._step(
                z,
                timestep,
                self.inverse_scheduler,
                conditioning,
            )

        return z, trace

    # Used for stage II.
    def _progressive_cycle(
        self,
        z0,
        depth,           # Invert (depth - skip) steps, then denoise depth steps.
        attacked_trace,  # Attacked trajectory to guide away from.
        conditioning,
    ):
        z = z0.clone()

        # Partial inversion.
        # If depth <= num_skip_inversion, this loop is empty.
        for i in range(depth - self.num_skip_inversion):
            timestep = self.inverse_scheduler.timesteps[i]
            next_timestep = (
                self.inverse_scheduler.timesteps[i + 1].item()
            )

            z = self._step(
                z,
                timestep,
                self.inverse_scheduler,
                conditioning,
            )

            # Guide away from the original attacked trajectory.
            if next_timestep in attacked_trace:
                # Use the next entry because tracing occurs before
                # each update. Both latents have now completed
                # the same inversion step.
                z_attacked = attacked_trace[next_timestep]

                z = z + self.guidance_strength * (z - z_attacked)

        # Denoise using the original starting index.
        start_index = self.num_inference_steps - depth

        for i in range(start_index, self.num_inference_steps):
            timestep = self.forward_scheduler.timesteps[i]

            z = self._step(
                z,
                timestep,
                self.forward_scheduler,
                conditioning,
            )

        return z

    @torch.no_grad()
    def run_defense(self, attacked_image):
        """Return the final defended zT for an attacked image."""
        # Encode the attacked image into its scaled, batched z0.
        attacked_z0 = self.pipe.imgs_to_latents(attacked_image).to(
            device=self.pipe.device,
            dtype=self.pipe.get_dtype(),
        )

        # Prepare the empty-prompt conditioning for SD or PixArt.
        conditioning = self.pipe.prepare_null_prompt_embeds(
            attacked_z0
        )

        self.inverse_scheduler.set_timesteps(
            self.num_inference_steps,
            device=attacked_z0.device,
        )
        self.forward_scheduler.set_timesteps(
            self.num_inference_steps,
            device=attacked_z0.device,
        )

        # Stage I: Record the standard inversion trajectory.
        _, attacked_trace = self._invert(
            attacked_z0,
            conditioning,
            store_trace=True,
        )

        # Stage II: Progressive Latent Refinement.
        defended_z0 = attacked_z0.clone()

        for depth in range(1, self.stopping_timestep + 1):
            defended_z0 = self._progressive_cycle(
                defended_z0,
                depth,
                attacked_trace,
                conditioning,
            )

        # Stage III: Final Inversion
        final_zT, _ = self._invert(
            defended_z0,
            conditioning,
        )
        return {
            "z0": defended_z0,
            "zT": final_zT,
        }