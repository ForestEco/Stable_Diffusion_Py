#Project derived from https://thepythoncode.com/article/control-generated-images-with-controlnet-with-huggingface

# $ pip install -qU xformers diffusers transformers accelerate
# $ pip install -qU  controlnet_aux
# $ pip install opencv-contrib-python

import os
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from tqdm import tqdm
from torch import autocast

# load the openpose model
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

# load the controlnet for openpose
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
# define stable diffusion pipeline with controlnet
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet,
                                                         safety_checker=None, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# enable efficient implementations using xformers for faster inference
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Directory containing the images
image_dir = r"D:\UNREAL ENGINE\Stable-Diffusion-Development\Marathon-Runner-KeyFrames"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Process each image in the directory
for image_file in tqdm(image_files):
    # Load the image
    image_path = os.path.join(image_dir, image_file)
    image_input = load_image(image_path)

    # Resize the image to 768x768
    image_input = image_input.resize((768, 768))

    # Run the openpose model on the image
    image_pose = openpose(image_input)

    # Save the pose image
    pose_image_path = os.path.join(image_dir, f"pose_{image_file}")
    image_pose.save(pose_image_path)

#Optional
# image_output = pipe("A pixelart character", image_pose, num_inference_steps=20).images[0]
# image_output
