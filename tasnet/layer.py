import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
from .param import TasNetParam


class Encoder(tf.keras.layers.Layer):
    """Encoder for mixture weight calculation (Lou et al., 2018).

    Attributes:
        U: Fully connected layer that learns N vectors with length L
           with `relu` activation.
        V: Fully connected layer that learns N vectors with lenght L
           with `sigmoid` activation.
        gating: Elementwise muliplication layer that multiply
                results of U and V elementwisely for `gating mechanism`.

        * N: Number of basis signals of the TasNet.
        * L: Length of each segment in input mixtures of the TasNet.
    """

    def __init__(self, param: TasNetParam, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.U = tf.keras.layers.Dense(units=param.N, activation='relu')
        self.V = tf.keras.layers.Dense(units=param.N, activation='sigmoid')
        self.gating = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        """Calculates mixture weight for each mixture segment.

        Args:
            mixture_segments:
                Input segments of the original mixture x(t).
                tf.Tensor of shape=(, K, L)

        Returns:
            mixture_weights:
                Calculated weights corresponding to the segments.
                tf.Tensor of shape=(, K, N)
        """
        return self.gating([self.U(mixture_segments), self.V(mixture_segments)])


class Decoder(tf.keras.layers.Layer):
    """Decoder for waveform reconstruction (Lou et al., 2018).

    Attributes:
        B: Fully connected layer that learns N basis signals of the input mixtures.

        * N: Number of basis signals of the TasNet.
        * L: Length of each segment in input mixtures of the TasNet.
    """

    def __init__(self, param: TasNetParam, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.B = tf.keras.layers.Dense(param.L)

    def call(self, source_weights):
        """Reconstructs waveform of sources from given weights.

        Args:
            source_weights:
                Weights for C clean sources.
                tf.Tensor of shape=(, C, K, L)

        Returns:
            estimated_sources:
                Reconstructed C sources by using `source_weights` and `B`.
                tf.Tensor of shape=(, C, K, N)
        """
        return self.B(source_weights)


class Separator(tf.keras.layers.Layer):
    """Separation network of the TasNet (Luo et al., 2018)

    In Separation Newtork, deep LSTM network is used. The directionality
    (uni-/bi-) of each LSTM layer is depend on the `causality` of the TasNet.
    Starting from the second LSTM layer, an identity skip connection is added
    between every two LSTEM layers to enhance the gradinet flow and accelerate
    the training process (p. 697). Separation network gets `mixture_weights`
    from the Encoder, and returns `sources_weights` to the Decoder.

    Attributes:
        layer_normalization: Layer normalization layer.
        C: Number of sources to separate.
        lstm1~6: Uni-/Bi-directional LSTM layer.
        skip_conntection: `Add` layer of the keras for `skip connection`.
    """

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
        # self.softmax = tf.keras.layers.Softmax(axis=-2)  # axis: C
        self.sigmoid = tf.keras.activations.sigmoid

        self.concat_weights = tf.keras.layers.concatenate
        self.reshape_weights = tf.keras.layers.Reshape(
            target_shape=[param.K, param.C, param.N])

        self.apply_mask = tf.keras.layers.Multiply()
        self.permute = tf.keras.layers.Permute([2, 1, 3])

    def call(self, mixture_weights):
        """Makes weights of C sources from mixture weights.

        Args:
            mixture_weights:
                Weights for the `mixture_segments`.
                tf.Tensor of shape=(, K, N)

        Returns:
            source_weights:
                Weights for C clean sources.
                tf.Tensor of shape=(, C, K, N)
        """

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
        # (, K, C, N) -> (, K, C, N)
        # source_masks = self.softmax(source_masks)
        source_masks = self.sigmoid(source_masks)

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
