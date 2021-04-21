import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
from .param import TasNetParam


class Encoder(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, name='Encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.U = tf.keras.layers.Conv1D(param.N, 1, activation='relu')
        self.V = tf.keras.layers.Conv1D(param.N, 1, activation='sigmoid')
        self.gating = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        # (, K, L) -> (, K, N)
        return self.gating([self.U(mixture_segments), self.V(mixture_segments)])


class Separator(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, name='Separation', **kwargs):
        super(Separator, self).__init__(name=name, **kwargs)

        self.layer_normalization = tf.keras.layers.LayerNormalization()
        self.C = param.C

        # deep LSTM network
        if param.causal:
            self.lstm1 = LSTM(param.H, return_sequences=True)
            self.lstm2 = LSTM(param.H, return_sequences=True)
            self.lstm3 = LSTM(param.H, return_sequences=True)
            self.lstm4 = LSTM(param.H, return_sequences=True)
            self.lstm5 = LSTM(param.H, return_sequences=True)
            self.lstm6 = LSTM(param.H, return_sequences=True)
        else:
            self.lstm1 = Bidirectional(LSTM(param.H, return_sequences=True))
            self.lstm2 = Bidirectional(LSTM(param.H, return_sequences=True))
            self.lstm3 = Bidirectional(LSTM(param.H, return_sequences=True))
            self.lstm4 = Bidirectional(LSTM(param.H, return_sequences=True))
            self.lstm5 = Bidirectional(LSTM(param.H, return_sequences=True))
            self.lstm6 = Bidirectional(LSTM(param.H, return_sequences=True))
        self.skip_connection = tf.keras.layers.Add()

        self.fc_layer = tf.keras.layers.Dense(param.N * param.C)
        self.make_mask = tf.keras.layers.Reshape([param.K, param.C, param.N])
        self.softmax = tf.keras.layers.Softmax(axis=-2)

        self.concat_weights = tf.keras.layers.concatenate  # function
        self.reshape_weights = tf.keras.layers.Reshape(
            [param.K, param.C, param.N])

        self.apply_mask = tf.keras.layers.Multiply()

    def call(self, mixture_weights):
        normalized_weights = self.layer_normalization(mixture_weights)

        lstm1_outputs = self.lstm1(normalized_weights)
        lstm2_outputs = self.lstm2(lstm1_outputs)
        lstm3_outputs = self.lstm3(lstm2_outputs)
        lstm4_outputs = self.lstm4(lstm3_outputs)
        lstm4_outputs = self.skip_connection([lstm2_outputs, lstm4_outputs])
        lstm5_outputs = self.lstm5(lstm4_outputs)
        lstm6_outputs = self.lstm6(lstm5_outputs)
        lstm6_outputs = self.skip_connection([lstm4_outputs, lstm6_outputs])

        # -> (, K, N*C)
        fc_outputs = self.fc_layer(lstm6_outputs)
        # (, K, N*C) -> (, K, C, N)
        source_masks = self.softmax(self.make_mask(fc_outputs))

        # (, K, N) -> (, K, N*C)
        mixture_weights = self.concat_weights(
            [mixture_weights for _ in range(self.C)], axis=-1)
        # (, K, N*C) -> (, K, C, N)
        mixture_weights = self.reshape_weights(mixture_weights)

        return self.apply_mask([mixture_weights, source_masks])


class Decoder(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decode = tf.keras.layers.Dense(param.L)
        self.permute = tf.keras.layers.Permute([2, 1, 3])

    def call(self, source_weights):
        # (, K, C, N) -> (, C, K, L)
        return self.permute(self.decode(source_weights))
