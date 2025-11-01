from src.vsg.handlers import FactualHandler, NvilaHandler


class NvilaFactualPipeline:
    def __init__(self):
        self.i_handler = NvilaHandler()
        self.sg_handler = FactualHandler()

    def image2text(self, image_path):
        return self.i_handler.frame2text(image_path)
    
    def text2triplets(self, text):
        return self.sg_handler.extract_triplets(text)
