import os
import pandas as pd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import FastText

class Embedding:

    MODEL_SAVED_DIR = "saved_model/fasttext.model"

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.corpus = dataset["TITLE"] + dataset["TEXTCONTENT"]

        # if os.path.isfile(self.MODEL_SAVED_DIR):
        #     self.fasttext = FastText.load(self.MODEL_SAVED_DIR)
        # else:
        self.fasttext = FastText(size=100, window=3, min_count=1)

    def extracte(self) -> None:
        self.extractor = WordExtractor()
        self.extractor.train(self.corpus)
        self.words = self.extractor.extract()
        self.cohesion_score = {word:score.cohesion_forward for word, score in self.words.items()}
        self.tokenizer = LTokenizer(scores=self.cohesion_score)

    def tokenize(self) -> pd.DataFrame:
        self.dataset["TITLE"] = self.dataset["TITLE"].apply(lambda text: self.tokenizer.tokenize(text))
        self.dataset["TEXTCONTENT"] = self.dataset["TEXTCONTENT"].apply(lambda text: self.tokenizer.tokenize(text))
        return self.dataset

    def train(self) -> None:
        self.fasttext.build_vocab(sentences=self.corpus)
        self.fasttext.train(sentences=self.corpus, total_examples=len(self.corpus), epochs=10)
        self.fasttext.save(self.MODEL_SAVED_DIR)

    def dataset_to_embedding(self) -> pd.DataFrame:
        pass

    def embedding_to_dataset(self) -> pd.DataFrame:
        pass

if __name__ == "__main__":
    # tokenize
    dataset = pd.read_json("data/20200101.json")    
    embedding = Embedding(dataset)
    embedding.extracte()
    tokenized = embedding.tokenize()
    print(tokenized["TITLE"][5])

    # fasttext
    embedding.train()
    print(embedding.fasttext["승리"])