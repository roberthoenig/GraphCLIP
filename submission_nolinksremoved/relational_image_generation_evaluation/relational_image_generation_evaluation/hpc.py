import torch
import clip
import os
from PIL import Image
from .download_weights import download_weights
from .evaluate_model import get_free_gpu

class HumanPreferenceScore:
    """
    Compute human preference score from text and images.
    """

    def __init__(self, device='auto'):
        self.weights_name = "hpc.pt"
        weight_path = os.path.join(os.path.dirname(__file__), 'data', self.weights_name)
        if not os.path.exists(weight_path):
            download_weights(self.weights_name)
            print(f'Downloaded weights for the HumanPreferenceScore Model to {weight_path}')
        else:
            print(f"File path {weight_path} already exists.")
        self.device = device if device != 'auto' else get_free_gpu()
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        params = torch.load(
            weight_path,
            map_location=self.device,
        )["state_dict"]
        self.model.load_state_dict(params)

    def compute_from_paths(self, text, image_paths):
        """
        ::param text: str
        ::param image_paths: list of str
        ::return: torch.Tensor
        """
        images = [Image.open(image_path) for image_path in image_paths]
        return self.compute(text, images)

    @torch.no_grad()
    def compute(self, text, images):
        """
        ::param text: str
        ::param images: list of PIL.Image
        ::return: torch.Tensor
        """
        processed_images = []
        for image in images:
            processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            processed_images.append(processed_image)
        processed_images = torch.cat(processed_images, dim=0)
        text = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            hps = image_features @ text_features.T

        return hps
