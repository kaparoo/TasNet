# Copyright (c) 2021 Jaewoo Park. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import math


class SDR(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = tf.keras.backend.epsilon()

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) /
                                (tf.norm(s) + self.eps) + self.eps) / math.log(10.0)
