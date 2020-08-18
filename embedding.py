import pandas as pd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

class Embedding:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.corpus = dataset["TEXTCONTENT"] + dataset["TITLE"]

    def extracte(self):
        self.extractor = WordExtractor()
        self.extractor.train(self.corpus)
        self.words = self.extractor.extract()

    def tokenize(self, sentence:str) -> list:
        self.tokenizer = LTokenizer(scores=self.words)
        return self.tokenizer.tokenize(sentence)

if __name__ == "__main__":
    dataset = pd.read_json("data/20200101.json")    
    print(dataset["TITLE"][0])
    embedding = Embedding(dataset)
    embedding.extracte()
    print(embedding.tokenize(dataset["TITLE"][0]))
