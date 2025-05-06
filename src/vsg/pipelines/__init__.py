from src.vsg.pipelines.gpt import GptPipeline
from src.vsg.pipelines.nvila_gpt import NvilaGptPipeline
from src.vsg.pipelines.nvida_factual import NvilaFactualPipeline


set_pipeline = {
    "nvila_factual": NvilaFactualPipeline,
    "nvila_gpt": NvilaGptPipeline,
    "gpt": GptPipeline
}
