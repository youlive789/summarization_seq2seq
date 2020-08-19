import os
import numpy as np
import pandas as pd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import FastText

class Embedding:

    MODEL_SAVED_DIR = "saved_model/fasttext.model"

    def __init__(self, dataset:pd.DataFrame, word_train:bool):
        self.dataset = dataset
        self.corpus = dataset["TITLE"] + dataset["TEXTCONTENT"]

        if os.path.isfile(self.MODEL_SAVED_DIR) or word_train == False:
            self.fasttext = FastText.load(self.MODEL_SAVED_DIR)
        else:
            self.fasttext = FastText(size=100, window=3, min_count=1)
            self._extracte()
            self._tokenize()
            self._train()

        self.idx_dict = dict(zip(np.arange(4, len(self.fasttext.wv.syn0) + 4), self.fasttext.wv.index2word))

    def _extracte(self) -> None:
        self.extractor = WordExtractor()
        self.extractor.train(self.corpus)
        self.words = self.extractor.extract()
        self.cohesion_score = {word:score.cohesion_forward for word, score in self.words.items()}
        self.tokenizer = LTokenizer(scores=self.cohesion_score)

    def _tokenize(self) -> pd.DataFrame:
        self.dataset["TITLE"] = self.dataset["TITLE"].apply(lambda text: self.tokenizer.tokenize(text))
        self.dataset["TEXTCONTENT"] = self.dataset["TEXTCONTENT"].apply(lambda text: self.tokenizer.tokenize(text))

    def _train(self) -> None:
        self.fasttext.build_vocab(sentences=self.corpus)
        self.fasttext.train(sentences=self.corpus, total_examples=len(self.corpus), epochs=10)
        self.fasttext.save(self.MODEL_SAVED_DIR)

    def dataset_to_embedding(self) -> pd.DataFrame:
        self.dataset["TITLE_IDX"] = self.dataset["TITLE"].apply(lambda tokenized: [self._word_to_idx(token) for token in tokenized])
        self.dataset["TITLE"] = self.dataset["TITLE"].apply(lambda tokenized: [self.fasttext[token] for token in tokenized])
        self.dataset["TEXTCONTENT"] = self.dataset["TEXTCONTENT"].apply(lambda tokenized: [self.fasttext[token] for token in tokenized])
        return self.dataset 

    def embedding_to_sentence(self, target: list or np.array) -> list:
        return [self._vec_to_word(vector) for vector in target]

    def _vec_to_word(self, vector) -> str:
        if np.array_equal(vector, np.eye(100, dtype=np.float32)[0]): return '<PAD>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[1]): return '<STA>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[2]): return '<EOS>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[3]): return '<UNK>'
        return self.fasttext.wv.similar_by_vector(vector)[0][0]

    def _word_to_vec(self, word) -> np.array:
        try :
            if word == '<PAD>': return np.eye(100, dtype=np.float32)[0]
            elif word == '<STA>': return np.eye(100, dtype=np.float32)[1]
            elif word == '<EOS>': return np.eye(100, dtype=np.float32)[2]
            elif word == '<UNK>': return np.eye(100, dtype=np.float32)[3]
            return self.fasttext.wv.word_vec(word)
        except :
            return np.eye(100, dtype=np.float32)[3]
    
    def _word_to_idx(self, word) -> int:      
        try :
            return list(self.idx_dict.keys())[list(self.idx_dict.values()).index(word)]
        except :
            return 3
    
    def _idx_to_word(self, idx) -> str:
        return self.idx_dict[idx]

if __name__ == "__main__":
    # tokenize
    dataset = pd.read_json("data/20200101.json")    
    embedding = Embedding(dataset=dataset, word_train=False)

    # embedding
    dataset = embedding.dataset_to_embedding()

    # reverse
    print(dataset["TITLE"][5])
    print(embedding.embedding_to_sentence(embedding.dataset["TITLE"][5]))
    