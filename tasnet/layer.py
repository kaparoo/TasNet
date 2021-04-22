# Copyright (c) 2021 Jaewoo Park. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
from .param import TasNetParam


class Encoder(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.U = tf.keras.layers.Conv1D(filters=param.N,
                                        kernel_size=1,
                                        activation='relu')
        self.V = tf.keras.layers.Conv1D(filters=param.N,
                                        kernel_size=1,
                                        activation='sigmoid')
        self.gating = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        # (, K, L) -> (, K, N)
        return self.gating([self.U(mixture_segments), self.V(mixture_segments)])


class Decoder(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.B = tf.keras.layers.Dense(param.L)

    def call(self, source_weights):
        # (, C, K, N) -> (, C, K, L)
        return self.B(source_weights)


class Separator(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(Separator, self).__init__(name='Separation', **kwargs)

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
        self.reshape_mask = tf.keras.layers.Reshape(
            target_shape=[param.K, param.C, param.N])
        self.softmax = tf.keras.layers.Softmax(axis=-2)  # axis: C

        self.concat_weights = tf.keras.layers.concatenate
        self.reshape_weights = tf.keras.layers.Reshape(
            target_shape=[param.K, param.C, param.N])

        self.apply_mask = tf.keras.layers.Multiply()
        self.permute = tf.keras.layers.Permute([2, 1, 3])

    def call(self, mixture_weights):
        # (, K, N) -> (, K, N)
        normalized_weights = self.layer_normalization(mixture_weights)

        # (, K, N) -> (, K, H)
        lstm1_outputs = self.lstm1(normalized_weights)
        lstm2_outputs = self.lstm2(lstm1_outputs)
        lstm3_outputs = self.lstm3(lstm2_outputs)
        lstm4_outputs = self.lstm4(lstm3_outputs)
        lstm4_outputs = self.skip_connection([lstm2_outputs, lstm4_outputs])
        lstm5_outputs = self.lstm5(lstm4_outputs)
        lstm6_outputs = self.lstm6(lstm5_outputs)
        lstm6_outputs = self.skip_connection([lstm4_outputs, lstm6_outputs])

        # (, K, H) -> (, K, C*N)
        fc_outputs = self.fc_layer(lstm6_outputs)

        # (, K, C*N) -> (, K, C, N)
        source_masks = self.reshape_mask(fc_outputs)
        # (, K, C, N) -> (, K, N*c)
        source_masks = self.softmax(fc_outputs)
        # (, K, C*N) -> (, K, C, N)
        source_masks = self.reshape_mask(fc_outputs)

        # (, K, N) -> (, K, C*N)
        mixture_weights = self.concat_weights(
            [mixture_weights for _ in range(self.C)], axis=-1)
        # (, K, C*N) -> (, K, C, N)
        mixture_weights = self.reshape_weights(mixture_weights)

        # (, K, C, N), (, K, C, N) -> (, K, C, N)
        source_weights = self.apply_mask([mixture_weights, source_masks])
        # (, K, C, N) -> (, C, K, N)
        source_weights = self.permute(source_weights)

        return source_weights
