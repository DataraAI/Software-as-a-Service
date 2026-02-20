# Imports


class AnnotationEngine():
    # Might use "video" instead of "frame_list"
    def __init__(self, frame_list):
        # Create variables
        self.frame_list = frame_list

        # Call preAnnotation
        self.preAnnotation()

    # Exocentric frames -> modified frames
    # Inpainting to remove human, most likely through ROSE
    def preAnnotation(self):
        # preAnnotation steps go here
        # ...

        # Then call inAnnotation
        self.inAnnotation()

    # Modified frames -> ego frames
    def inAnnotation(self):
        # inAnnotation steps go here
        # ...

        # Then call postAnnotation
        self.postAnnotation()

    # Egocentric frames -> extracting details about the frames
    def postAnnotation(self):
        # postAnnotation steps go here
        # ...

        # Then return ego + annotations
        return


# Example use case
annotation_engine = AnnotationEngine(frame_list)



