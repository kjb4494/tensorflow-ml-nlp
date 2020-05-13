from keras.layers import Layer
from keras import backend as K


class NonMasking(Layer):
    def __init__(self, **kwargs):
        super(NonMasking, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerStart(Layer):
    W = None

    def __init__(self, seq_len, **kwargs):
        super(LayerStart, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(
            name='kernel',
            shape=(input_shape[2], 2),
            initializer='uniform',
            trainable=True
        )
        super(LayerStart, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = K.reshape(inputs, shape=(-1, self.seq_len, K.shape(inputs)[2]))
        x = K.dot(x, self.W)
        x = K.permute_dimensions(x, (2, 0, 1))
        return K.softmax(x[0], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.seq_len


class LayerEnd(Layer):
    W = None

    def __init__(self, seq_len, **kwargs):
        super(LayerEnd, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(
            name='kernel',
            shape=(input_shape[2], 2),
            initializer='uniform',
            trainable=True
        )
        super(LayerEnd, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = K.reshape(inputs, shape=(-1, self.seq_len, K.shape(inputs)[2]))
        x = K.dot(x, self.W)
        x = K.permute_dimensions(x, (2, 0, 1))
        return K.softmax(x[1], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.seq_len
