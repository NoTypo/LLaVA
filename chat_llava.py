import os
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# Load model and tokenizer
model_path = "liuhaotian/llava-v1.5-7b"
kwargs = {
    "device_map": "auto",
    "load_in_4bit": True,
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
}

model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Load vision tower
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

# Function to preprocess image
def load_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

# Prepare conversation template
conv_mode = "v0_mmtag"
conv = conv_templates[conv_mode].copy()
roles = conv.roles

# Initialize conversation with an image and prompt
def initialize_conversation(image_url, prompt):
    image_tensor = load_image(image_url)
    inp = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{roles[0]}: {prompt}"
    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    return image_tensor

# Generate response from the model
def generate_response(image_tensor):
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_tensor, do_sample=True, temperature=0.2,
            max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria]
        )

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().rsplit('</s>', 1)[0]
    conv.messages[-1][-1] = output
    return output

# Main conversation loop
def chat():
    image_url = input("Enter image URL: ")
    prompt = input("Enter your prompt: ")
    image_tensor = initialize_conversation(image_url, prompt)

    while True:
        response = generate_response(image_tensor)
        print(f"Model: {response}")
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        conv.append_message(roles[0], user_input)
        conv.append_message(roles[1], None)

chat()
