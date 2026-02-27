import argparse
import image_prompt_tool


class AnnotationEngine_Image_Only:
    def __init__(self, ego_prompt, imageURL, container_name):
        self.ego_prompt = ego_prompt
        self.imageURL = imageURL
        self.container_name = container_name

    def run(self):
        self.convert_to_ego()
        self.invoke_vlm()

    def convert_to_ego(self):
        self.egoImageFilepath = image_prompt_tool.main([
            "--ego_prompt", self.ego_prompt,
            "--imageURL", self.imageURL,
            "--container_name", self.container_name])

    def invoke_vlm(self):
        # TODO: add script to call the Qwen VLM models
        # Don't use the self.ego_prompt variable, that's the prompt to rotate and remove human
        # Add in your own custom prompts, then pass them into the VLM with the self.egoImageFilepath

        # Should have a self.output_json_path variable of some kind,
        # i.e. saving the output of the VLM
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego_prompt", type=str, help="input ego_prompt")
    parser.add_argument("--imageURL", type=str, help="input imageURL")
    parser.add_argument("--container_name", type=str, help="Azure Blob's container name")

    args = parser.parse_args()
    ego_prompt = args.ego_prompt
    imageURL = args.imageURL
    container_name = args.container_name

    annotation_engine = AnnotationEngine_Image_Only(ego_prompt, imageURL, container_name)
    annotation_engine.run()

    # Using print b/c that's how DaaS can pick up the output of image + annotation paths
    print(annotation_engine.egoImageFilepath)
    print(annotation_engine.output_json_path)


if __name__ == "__main__":
    main()

