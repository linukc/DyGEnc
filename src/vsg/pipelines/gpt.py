from src.vsg.handlers import GptHandler


class GptPipeline:
    def __init__(self):
        self.handler = GptHandler()

    def image2text(self, image_path):
        return self.handler.image2text(image_path)
    
    def text2triplets(self, text):
        return self.handler.text2triplets(text)
