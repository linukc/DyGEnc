from PIL import Image
from loguru import logger
from transformers import AutoModel


PROMPT = \
"""Please provide a caption for the attached image.
Focus on describing actions or spatial relationships between a person and objects with which he interacts.
If you see several instances of one type - describe them individually."""

class NvilaHandler:
    def __init__(self):
        logger.info("loading Nvila")
        model_path = "Efficient-Large-Model/NVILA-Lite-8B-hf-preview"
        kwargs = {
            "device_map": "cuda",
            "revision": "main",
        }
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)


    def frame2text(self, image_path):
        response = self.model.generate_content([
            Image.open(image_path),
            PROMPT
        ])
        return response
