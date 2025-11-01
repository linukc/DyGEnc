from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.vsg.handlers.utils import Triplet, TripletExtraction


class FactualHandler:
    def __init__(self, model_name="lizhuang144/flan-t5-base-VG-factual-sg"):
        """
        Initializes the FLAN-T5 model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained model to load.
        """
        logger.info("loading Factual model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def extract_triplets(self, input_text, max_length=1000, num_beams=3):
        """
        Extracts factual triplets from the input text using a sequence-to-sequence model.

        Args:
            input_text (str): The input text to process.
            max_length (int): The maximum length of the tokenized input and output sequences.
            num_beams (int): The number of beams to use in beam search decoding.

        Returns:
            TripletExtraction: A structured representation containing extracted triplets.
        """
        text = self.tokenizer(
            input_text, max_length=max_length, return_tensors="pt", truncation=True
        )

        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=False,
        )

        output = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        triplets = output[1:-1].split("), (")
        parsed_triplets = []
        for triplet in triplets:
            t = triplet.split(", ")
            parsed_triplets.append(Triplet(source=t[0], edge=t[1], target=t[2]))

        return TripletExtraction(triplets=parsed_triplets)
