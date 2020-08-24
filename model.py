import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *

"""
reference: https://heung-bae-lee.github.io/2020/01/22/deep_learning_11/
"""

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = LSTM(128, return_state=True)

    def call(self, x, training=False, mask=None):
        _, h, c = self.lstm(x)
        return h, c

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = LSTM(128, return_state=True, return_sequences=True)
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x, h, c = inputs
        x, h, c = self.lstm(x, initial_state=[h, c])
        return self.dense(x), h, c
        
class Seq2Seq(tf.keras.Model):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def call(self, inputs, training=False, mask=None):
        if training is True:
            x, y = inputs
            h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c))
            return y
        else:
            # 추론 중에는 디코더에 패딩으로 채운 값을 입력으로 줘야한다.
            x = inputs
            h, c = self.enc(x)

            y = tf.convert_to_tensor(np.zeros((1, 10, 100)), dtype=tf.float32)

            y, h, c = self.dec((y, h, c))
            y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)

            return tf.reshape(y, (1, 10))

@tf.function
def train_step(model, encoder_input, decoder_input, decoder_output, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model([encoder_input, decoder_input], training=True)
        loss = loss_object(decoder_output, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(decoder_output, predictions)

@tf.function
def test_step(model, inputs):
    return model(inputs, training=False)

if __name__ == "__main__":
    encoder_input = np.zeros((1, 32, 100))
    decoder_input = np.zeros((1, 10, 100))
    decoder_output = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    model = Seq2Seq()

    loss_obejct = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(50):
        train_step(model, encoder_input, decoder_input, decoder_output, loss_obejct, optimizer, train_loss, train_accuracy)
        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))

    prediction = test_step(model, encoder_input)
    print("결과 :", prediction)

    

