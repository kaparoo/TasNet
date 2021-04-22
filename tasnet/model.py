import tensorflow as tf
from .param import TasNetParam
from .layer import Encoder, Decoder, Separator
from .loss import SDR


class TasNet(tf.keras.Model):
    def __init__(self, param: TasNetParam, **kwargs):
        super(TasNet, self).__init__(**kwargs)
        self.param = param
        self.encode = Encoder(param)
        self.separate = Separator(param)
        self.decode = Decoder(param)

    def call(self, mixture_segments):
        mixture_weights = self.encode(mixture_segments)
        source_weights = self.separate(mixture_weights)
        estimated_sources = self.decode(source_weights)
        return estimated_sources  # (, K, L) -> (, C, K, L)

    @staticmethod
    def make(param: TasNetParam):
        model = TasNet(param)
        model.compile(optimizer='adam', loss=SDR())
        model.build(input_shape=(None, param.K, param.L))
        return model
