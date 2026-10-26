# Imprint attacks 
from abc import ABC, abstractmethod
from typing import Callable, Sequence, Type
import torch
import tqdm
from PIL import Image
from diffusers import DiffusionPipeline
from pipe.attack_pipe import DiffPipe, invert_image_with_attack_pipe, pixel_to_latent


class ImprintAttack(ABC):
    def __init__(
        self,
        inverse_scheduler,
        attack_pipe: DiffusionPipeline,
        image: Image.Image,
        reference_image: Image.Image,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        learning_rate: float = 1e-2,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        num_inference_steps_attacker: int = 50,
        steps: int = 151,
        save_steps: Sequence[int] = (50, 100, 150),
    ):
        self.attack_pipe = attack_pipe
        self.inverse_scheduler = inverse_scheduler

        self.image = image # Image we wish to perform optimization on - watermarked in removal case and cover image in forgery case
        self.reference_image = reference_image # Reference image for the attack - watermarked image

        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls

        self.num_inference_steps_attacker = num_inference_steps_attacker
        self.steps = steps
        self.save_steps = tuple(save_steps)

    def prepare_attack(self):
        # Initialize the latent that we will modify.
        z0_original = pixel_to_latent(self.image, self.attack_pipe)
        self.z0 = torch.nn.Parameter(z0_original.detach().clone())

        self.optimizer = self.optimizer_cls(
            [self.z0],
            lr=self.learning_rate,
        )

        with torch.no_grad():
            self.reference_zT = invert_image_with_attack_pipe(
                pipe=self.attack_pipe,
                image=self.reference_image,
                scheduler=self.inverse_scheduler,
                num_inference_steps=self.num_inference_steps_attacker,
                resolution=self.image.width,
            ).detach()

        # Differentiable diffpipe for attack
        self.diffpipe = DiffPipe(
            self.attack_pipe,
            scheduler=self.inverse_scheduler,
            device=self.attack_pipe.device,
        )

    @abstractmethod
    def compute_loss(self, inverted_latent: torch.Tensor) -> torch.Tensor:
        """Return the objective minimized by the optimizer."""
        pass

    def run(self):
        self.prepare_attack()

        saved_latents = {}
        loss_history = []

        for step in tqdm.tqdm(range(self.steps)):
            self.optimizer.zero_grad()

            inverted_latent = self.diffpipe(
                self.z0,
                "",
                guidance_scale=1.0,
                num_inference_steps=self.num_inference_steps_attacker,
            )

            loss = self.compute_loss(inverted_latent)

            loss.backward()
            self.optimizer.step()

            loss_history.append(loss.detach().item())

            if step in self.save_steps:
                saved_latents[step] = self.z0.detach().clone()

        return {
            "final_latent": self.z0.detach().clone(),
            "saved_latents": saved_latents,
            "loss_history": loss_history,
        }


class ImprintRemovalAttack(ImprintAttack):
    def compute_loss(self, inverted_latent):
        return self.loss_fn(inverted_latent, -self.reference_zT)


class ImprintForgeryAttack(ImprintAttack):
    def compute_loss(self, inverted_latent):
        return self.loss_fn(inverted_latent, self.reference_zT)
 