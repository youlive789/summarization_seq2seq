import pandas as pd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

class Embedding:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        # self.fasttext = FastText()

    def tokenize(self, sentence:str) -> list:
        self.tokenizer = LTokenizer()
        return self.tokenizer.tokenize(sentence)

if __name__ == "__main__":
    dataset = pd.read_json("data/20200101.json")    
    embedding = Embedding(dataset)

    # Tokenize
    dataset["TITLE"] = dataset["TITLE"].apply(lambda text: embedding.tokenize(text))
    dataset["TEXTCONTENT"] = dataset["TEXTCONTENT"].apply(lambda text: embedding.tokenize(text))    
    print(dataset["TITLE"][0])
    print()
    print(dataset["TITLE"][3])
    print(dataset["TEXTCONTENT"][3])

    # Fasttext embedding