# Imports

import argparse

import DataraAI_segmentation


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
        # preAnnotation steps go here
        human_masks = DataraAI_segmentation.mask_generation(self.video_path)
        # rose inpainting will take in video path and human masks (or similar)

    # Modified frames -> ego frames
    def inAnnotation(self):
        # inAnnotation steps go here
        # ...

    # Egocentric frames -> extracting details about the frames
    def postAnnotation(self):
        # postAnnotation steps go here
        # ...

        # Then return ego + annotations
        return


# Example use case
annotation_engine = AnnotationEngine(video_path)
annotation_engine.run()


