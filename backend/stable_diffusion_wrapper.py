from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

class StableDiffusionWrapper:
    def __init__(self) -> None:
        repo_id = "Linaqruf/anything-v3.0"
        pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16
        )
        

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

            
    def generate_images(self, text_prompt: str, num_images: int):
        prompt = [text_prompt] * num_images
        images = self.pipe(prompt, num_inference_steps=10).images
        return images
