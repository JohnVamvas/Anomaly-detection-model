from keras.layers import Layer
from keras import backend as K

class EtaLayer(Layer):

    def __init__(self, lam=0.1, trainable = True, **kwargs):
        super(EtaLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.lam = lam
        self.trainable = trainable

    def build(self, input_shape):
        self.lam_factor = K.variable(self.lam, dtype=K.floatx(), name='lam_factor')
        if self.trainable:
            self._trainable_weights.append(self.lam_factor)
        super(EtaLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        return K.sign(inputs) * K.maximum(K.abs(inputs) - self.lam_factor, 0)

    def get_config(self):
        config = {'lam': self.get_weights()[0] if self.trainable else self.lam,
                  'trainable': self.trainable}
        base_config = super(EtaLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

