import tensorflow as tf

class PilotModel(tf.keras.Model):
    def __init__(self):
        super(PilotModel, self).__init__(name="PilotModel")