from gradio_client import Client, handle_file
import argparse
import shutil
import os


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="input dataset name")
parser.add_argument("--frame_id", type=int, help="enter the frame number you want to use")
parser.add_argument(
    "--view_option",
    type=str,
    help="select which new ego view you want produced after the base ego view is done",
    choices=["base", "rotate_left", "rotate_right", "top_down", "low_angle"],
    default="base"
)
parser.add_argument(
    "--custom_prompt",
    type=str,
    help="enter in a custom prompt for the VLM",
    default="Create an image in the car assembler's perspective, and remove the person."
)


args = parser.parse_args()
dataset_name = args.dataset_name
frame_id = str(args.frame_id)
view_option = args.view_option
custom_prompt = args.custom_prompt


dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_list", dataset_name)

if not dataset_path:
    print(f"The dataset path '{dataset_path}' does not exist.")
    exit()

orig_path = os.path.join(dataset_path, "orig")

if not orig_path:
    print(f"The original images path '{orig_path}' does not exist.")
    exit()


ego_path = os.path.join(dataset_path, "egos")
os.makedirs(ego_path, exist_ok=True)



client = Client("tori29umai/Qwen-Image-2509-MultipleAngles")

if view_option == "base":
    image = handle_file(os.path.join(orig_path, f"{dataset_name}_{frame_id}.png"))
    dropdown_value_cn="__custom__"
    custom_cn = custom_prompt
    # custom_cn = "Create an image in the car assembler's perspective, and remove the person."
elif view_option == "rotate_left":
    image = handle_file(os.path.join(ego_path, f"{dataset_name}_{frame_id}_ego_base.png"))
    dropdown_value_cn = "镜头方向左回转45度"
    custom_cn = ""
elif view_option == "rotate_right":
    image = handle_file(os.path.join(ego_path, f"{dataset_name}_{frame_id}_ego_base.png"))
    dropdown_value_cn = "镜头向右回转45度"
    custom_cn = ""
elif view_option == "top_down":
    image = handle_file(os.path.join(ego_path, f"{dataset_name}_{frame_id}_ego_base.png"))
    dropdown_value_cn = "将镜头转为俯视"
    custom_cn = ""
elif view_option == "low_angle":
    image = handle_file(os.path.join(ego_path, f"{dataset_name}_{frame_id}_ego_base.png"))
    dropdown_value_cn = "将镜头转为仰视"
    custom_cn = ""

output_path, _ = client.predict(
    image=image,
    dropdown_value_cn=dropdown_value_cn,
    custom_cn=custom_cn,
    extra_prompt="",
    seed=0,
    randomize_seed=True,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    lang="en",
    api_name="/generate_from_dropdown"
)

# Define the source path of the file
source_file = output_path

# Define the destination path (can be a directory or a new file path)
destination_path = os.path.join(ego_path, f"{dataset_name}_{frame_id}_ego_{view_option}.png")

# Move the file
try:
    shutil.move(source_file, destination_path)
    # print(f"File '{source_file}' moved successfully to '{destination_path}'")
    print(f"Successfully created ego image at {destination_path}")
except FileNotFoundError:
    print(f"Error: Source file '{source_file}' not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

