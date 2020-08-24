import os
import pickle
import numpy as np
import pandas as pd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import FastText

class Embedding:

    MODEL_SAVED_DIR = "saved_model/fasttext.model"
    TOKENIZER_SAVED_DIR = "saved_model\\tokenizer.pkl"

    def __init__(self, dataset:pd.DataFrame, word_train:bool):
        self.dataset = dataset
        self.corpus = dataset["TITLE"] + dataset["TEXTCONTENT"]

        if word_train == False:
            self.fasttext = FastText.load(self.MODEL_SAVED_DIR)
            self._load_tokenizer()
            self._tokenize()
        else:
            self._extracte()
            self._tokenize()
            self._save_tokenizer()
            self._train()

        self.idx_word_dict = dict(zip(np.arange(4, len(self.fasttext.wv.vectors) + 4), self.fasttext.wv.index2word))
        self.idx_word_dict[0] = '<PAD>'
        self.idx_word_dict[1] = '<STA>'
        self.idx_word_dict[2] = '<EOS>'
        self.idx_word_dict[3] = '<UNK>'

    def _extracte(self) -> None:
        self.extractor = WordExtractor()
        self.extractor.train(self.corpus)
        self.words = self.extractor.extract()
        self.cohesion_score = {word:score.cohesion_forward for word, score in self.words.items()}
        self.tokenizer = LTokenizer(scores=self.cohesion_score)

    def _tokenize(self) -> pd.DataFrame:
        self.corpus = self.corpus.apply(lambda text : self.tokenizer.tokenize(text))
        self.dataset["TITLE"] = self.dataset["TITLE"].apply(lambda text: self.tokenizer.tokenize(text))
        self.dataset["TEXTCONTENT"] = self.dataset["TEXTCONTENT"].apply(lambda text: self.tokenizer.tokenize(text))

    def _save_tokenizer(self) -> None:
        with open(self.TOKENIZER_SAVED_DIR, "wb") as f:
            pickle.dump(self.tokenizer, f, pickle.HIGHEST_PROTOCOL)

    def _load_tokenizer(self) -> None:
        with open(self.TOKENIZER_SAVED_DIR, "rb") as f:
            self.tokenizer = pickle.load(f)

    def _train(self) -> None:
        self.fasttext = FastText(sentences=self.corpus, size=100, window=5, min_count=1, iter=100)
        self.fasttext.save(self.MODEL_SAVED_DIR)

    def dataset_to_embedding(self) -> pd.DataFrame:
        self.dataset["TITLE_IDX"] = self.dataset["TITLE"].apply(self._sentence_length_fix, args=[10])
        self.dataset["TITLE"] = self.dataset["TITLE"].apply(self._sentence_length_fix, args=[10])
        self.dataset["TEXTCONTENT"] = self.dataset["TEXTCONTENT"].apply(self._sentence_length_fix, args=[32])

        for index, value in self.dataset["TITLE_IDX"].iteritems():
            assert len(value) == 10

        for index, value in self.dataset["TITLE"].iteritems():
            assert len(value) == 10

        for index, value in self.dataset["TEXTCONTENT"].iteritems():
            assert len(value) == 32

        self.dataset["TITLE_IDX"] = self.dataset["TITLE_IDX"].apply(lambda tokenized: np.array([self._word_to_idx(token) for token in tokenized]))
        self.dataset["TITLE"] = self.dataset["TITLE"].apply(lambda tokenized: np.array([self._word_to_vec(token) for token in tokenized]))
        self.dataset["TEXTCONTENT"] = self.dataset["TEXTCONTENT"].apply(lambda tokenized: np.array([self._word_to_vec(token) for token in tokenized]))
        
        return self.dataset 

    def embedding_to_sentence(self, target: list or np.array) -> list:
        return [self._vec_to_word(vector) for vector in target]

    def _sentence_length_fix(self, sentence: list or np.array, length: int) -> list or np.array:
        sentence_length = len(sentence)
        if sentence_length < length:
            while len(sentence) < length:
                sentence.append('<PAD>')
        elif sentence_length > length:
            sentence = sentence[:length]
        return sentence

    def _vec_to_word(self, vector) -> str:
        if np.array_equal(vector, np.eye(100, dtype=np.float32)[0]):   return '<PAD>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[1]): return '<STA>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[2]): return '<EOS>'
        elif np.array_equal(vector, np.eye(100, dtype=np.float32)[3]): return '<UNK>'
        return self.fasttext.wv.most_similar(positive=[vector], topn=1)[0][0]

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
            return list(self.idx_word_dict.keys())[list(self.idx_word_dict.values()).index(word)]
        except :
            return 3
    
    def _idx_to_word(self, idx) -> str:
        return self.idx_word_dict[idx]

if __name__ == "__main__":
    # tokenize
    dataset = pd.read_json("data/20200101.json")    
    embedding = Embedding(dataset=dataset, word_train=False)

    # embedding
    dataset = embedding.dataset_to_embedding()

    # reverse
    print(embedding.embedding_to_sentence(dataset["TITLE"][0]))
    