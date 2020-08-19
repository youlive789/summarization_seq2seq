import numpy as np
import pandas as pd
from typing import Union
from embedding import Embedding
from sklearn.model_selection import train_test_split

class SummarizationDataset:

    def __init__(self, dataset:pd.DataFrame, word_train:bool):
        self.embedding = Embedding(dataset, word_train)
        self.dataset = self.embedding.dataset_to_embedding()

    def get_embedded_dataset(self) -> Union[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(self.dataset, test_size=0.33)
        return train, test

if __name__ == "__main__":
    dataset = pd.read_json("data/20200101.json")
    sd = SummarizationDataset(dataset, True)
    train, test = sd.get_embedded_dataset()