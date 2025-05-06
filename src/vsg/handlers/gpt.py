import base64

from openai import OpenAI
from dotenv import dotenv_values

from src.vsg.handlers.utils import TripletExtractionGPT


ImageSystemPrompt = \
"""You are a helpful assistant that captions images."""

ImageUserPrompt = \
"""Please provide a caption for the attached image.
Focus on describing actions or spatial relationships between a person and objects with which he interacts.
If you see several instances of one type - describe them individually."""

TripletsSystemPrompt = \
"""You are a helpful assistant that summaries text caption into a directed graph structure in the form of text triplets (source; edge; target).
Each triplet should capture either action or spatial relationship between a person and a unique single object, where:
(i) the source is always the person as a whole (without separating body parts), do not describe person;
(ii) the edge is a verb or preposition that describes action or spatial relationship between a person and a unique single object;
(iii) the target object is unique entity from the caption with its properties in two words: main attribute (adjective) and target (noun).
"""

TripletsUserPrompt = \
"""Here is the text caption"""

class GptHandler:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initializes the OpenAIHandler with API credentials and model settings.

        :param model: The name of the OpenAI model to use.
        """
        self.config = dotenv_values("gpt.env")
        self.api_key = self.config["OPENAI_API_KEY"]
        if not self.api_key:
            raise ValueError(
                """API key is required. Set OPENAI_API_KEY as
                an environment variable"""
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def create_prompt(self, text_response):
        """
        Creates a prompt for OpenAI API using a text response.

        :param text_response: The input text.
        :return: A structured prompt in dictionary format.
        """
        return [{"type": "text", "text": text_response}]

    def text2triplets(self, text_response):
        """
        Converts a text response into a structured set of triplets using OpenAI API.

        :param text_response: Input text to extract triplets from.
        :return: TripletExtraction object containing extracted triplets.
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": self.create_prompt(TripletsSystemPrompt),
                },
                {
                    "role": "user",
                    "content": self.create_prompt(f'{TripletsUserPrompt}: {text_response}'),
                }
            ],
            response_format=TripletExtractionGPT,
        )

        triplet_data = response.choices[0].message.parsed
        return triplet_data

    def encode_image(self, image_path):
        """
        Encodes an image file into a base64 string.

        :param image_path: Path to the image file.
        :return: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_prompt_with_image(self, image_path, text_response):
        """
        Creates a prompt that includes both an image and a text response.

        :param image_path: Path to the image file.
        :param text_response: Text input to complement the image.
        :return: A structured prompt including both text and image.
        """
        base64_image = self.encode_image(image_path)
        prompt = self.create_prompt(text_response)
        prompt.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                },
            }
        )
        return prompt

    def image2text(self, image_path):
        """
        Generates a text response based on an image and text prompt.

        :param image_path: Path to the image file.
        :param text_response: Text prompt for guiding the response.
        :return: The generated text response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": self.create_prompt(ImageSystemPrompt)
                },
                {
                    "role": "user",
                    "content": self.create_prompt_with_image(image_path, ImageUserPrompt),
                }
            ],
        )
        return response.choices[0].message.content
