from src.datasets.star import StarDataset
from src.datasets.agqa import AGQADataset


load_dataset = {
    "star": StarDataset,
    "agqa": AGQADataset,
}