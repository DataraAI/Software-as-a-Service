# Imports

import argparse
import os
from PIL import Image
import numpy as np

import sys
from pathlib import Path

# Add services/segmentation/src to path
seg_path = Path(__file__).parent.parent / "services" / "segmentation" / "src"
sys.path.append(str(seg_path))

import segmentation as DataraAI_segmentation

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="input MP4 video_path")

args = parser.parse_args()
video_path = args.video_path



class AnnotationEngine():
    def __init__(self, video_path):
        # Create variables
        self.video_path = video_path

    def run(self):
        self.preAnnotation()
        self.inAnnotation()
        self.postAnnotation()

    # Exocentric frames -> modified frames
    # Inpainting to remove human, most likely through ROSE
    def preAnnotation(self):
        # sam3 conda environment
        # os.system("conda activate sam3")

        outputs_per_frame = DataraAI_segmentation.mask_generation(self.video_path)

        # This will write human masks to ~/masks/mask_<frame id>.png
        # It should take in video input without changing fps
        i = 0
        os.makedirs(os.path.join(os.path.expanduser("~"), "masks"), exist_ok=True)
        for output in outputs_per_frame.values():
            human_binary_masks = output["out_binary_masks"]
            # Take only the first human, i.e. human_binary_masks[0]
            mask = human_binary_masks[0].astype(np.uint8) * 255
            mask_image = Image.fromarray(mask)
            mask_image.save(os.path.join(os.path.expanduser("~"), "masks") + "/mask_" + str(i) + '.png')
            i += 1

        # os.system("conda deactivate")
        # rose inpainting will take in video path and human masks (or similar)

    # Modified frames -> ego frames
    def inAnnotation(self):
        # ...
        return

    # Egocentric frames -> extracting details about the frames
    def postAnnotation(self):
        # ...

        # Then return ego + annotations
        return


# Example use case
annotation_engine = AnnotationEngine(video_path)
# annotation_engine.run()
annotation_engine.preAnnotation()


