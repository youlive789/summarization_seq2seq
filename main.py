import numpy as np
import pandas as pd
import tensorflow as tf

import json
from model import Seq2Seq, train_step, test_step
from embedding import Embedding
from data import SummarizationDataset

if __name__ == "__main__":
    
    dataset = pd.read_json("data/blog.json", encoding="utf-8")
    sd = SummarizationDataset(dataset=dataset, word_train=True)
    train, test = sd.get_embedded_dataset()

    encoder_input = np.array([x.reshape(32, 100) for x in train["TEXTCONTENT"]])
    decoder_input = np.array([x.reshape(10, 100) for x in train["TITLE"]])
    decoder_output = np.array([x.reshape(10) for x in train["TITLE_IDX"]])
    testset_input = np.array([x.reshape(32, 100) for x in test["TEXTCONTENT"]])

    vocab_length = len(sd.embedding.idx_word_dict)
    model = Seq2Seq(vocab_length = vocab_length)
    loss_obejct = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(50):
        train_step(model, encoder_input, decoder_input, decoder_output, loss_obejct, optimizer, train_loss, train_accuracy)
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))

    prediction = test_step(model, testset_input[0:1]).numpy().tolist()[0]
    print("요약된 문장 : ", [sd.embedding._idx_to_word(x) for x in prediction])