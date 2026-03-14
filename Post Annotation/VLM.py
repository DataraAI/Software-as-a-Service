# pip install git+https://github.com/huggingface/transformers accelerate
# pip install qwen-vl-utils[decord]==0.0.8

# Since the Lambda VM is aarch64, ignore the previous pip command and replace with this:
# pip install qwen-vl-utils


import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image1 = load_image("/content/datara_ai/backend/dataset_list/BMW_Front_Bumper/egos/BMW_Front_Bumper_1_ego_base.jpg")

import cv2
from google.colab.patches import cv2_imshow

cv2_imshow(cv2.imread("/content/datara_ai/backend/dataset_list/BMW_Front_Bumper/egos/BMW_Front_Bumper_1_ego_base.jpg", cv2.IMREAD_UNCHANGED))

#Load Qwen VLM Model

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


#Generate Text Function
# from qwen_vl_utils import load_image, load_video
from transformers.image_utils import load_image
from transformers.video_utils import load_video
from qwen_vl_utils import process_vision_info

def generate_text(model, processor, filepath, label, detail):
    # Get media type, image/video
    if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
        media_type = "image"
        media = load_image(filepath)
    elif filepath.endswith(".mp4") or filepath.endswith(".mov"):
        media_type = "video"
        media = load_video(filepath)
    else:
        raise ValueError("Unsupported media type")

    # Load prompt based off of the kind of detail requested
    prompt = "This is a " + ("successful" if label == "good" else "defective") + "welding job. "
    if detail == "condition":
        prompt += "What 1-2 word qualities satisfy this condition?"
    elif detail == "operation type":
        prompt += "What is its operation type?"
    elif detail == "sensor modalities":
        prompt += "What are the sensor modalities?"
    else:
        raise ValueError("Unsupported detail type")

    # Prepare the input
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: media,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [o for output in output_text for o in output.split("\\n")]

#Condition
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image1,
            },
            {
                "type": "text",
                "text": "Give me a list of one word qualities about this image that makes it a successful welding job."
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
# print(output_text)
for output in output_text:
    for o in output.split("\\n"):
        print(o)

#Operation Type
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image1,
            },
            {
                "type": "text",
                "text": "What is the operation type of the welding job in this image?"
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
# print(output_text)
for output in output_text:
    for o in output.split("\\n"):
        print(o)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image1,
            },
            {
                "type": "text",
                "text": "Detect the sensor modalities of the welding job in the image."
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
# print(output_text)
for output in output_text:
    for o in output.split("\\n"):
        print(o)