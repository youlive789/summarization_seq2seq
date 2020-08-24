import numpy as np
import pandas as pd
import tensorflow as tf

import json
from model import Seq2Seq, train_step, test_step
from embedding import Embedding
from data import SummarizationDataset

if __name__ == "__main__":
    
    dataset = pd.read_json("data/20200101.json")
    sd = SummarizationDataset(dataset, False)
    train, test = sd.get_embedded_dataset()

    encoder_input = np.array([x.reshape(32, 100) for x in train["TEXTCONTENT"]])
    decoder_input = np.array([x.reshape(10, 100) for x in train["TITLE"]])
    decoder_output = np.array([x.reshape(10) for x in train["TITLE_IDX"]])

    batch_size = 16
    batch_count = int(len(encoder_input) / batch_size)

    # 모델 구조 Dense layer가 10이아닌  
    # vocabulary size 로 구성해야함.

    model = Seq2Seq()
    loss_obejct = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(5):
        for idx in range(batch_count):
            train_step(model, encoder_input[idx:idx*batch_size], decoder_input[idx:idx*batch_size], decoder_output[idx:idx*batch_size], loss_obejct, optimizer, train_loss, train_accuracy)
            template = 'Epoch {}, Loss: {}, Accuracy: {}'
            print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))

    prediction = test_step(model, test["TEXTCONTENT"][0])
    print(prediction)