from src.datasets.star import StarDataset
from src.datasets.agqa import AGQADataset
from src.datasets.custom import UserDataset


load_dataset = {
    "star": StarDataset,
    "agqa": AGQADataset,
    "custom": UserDataset
}