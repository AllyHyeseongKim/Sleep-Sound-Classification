from typing import Any, Optional, Sequence, Tuple

import tensorflow as tf


class AudioClassifier(tf.keras.Model):

    def __init__(self, num_outputs: int,
                 frontend: Optional[tf.keras.Model] = None,
                 encoder: Optional[tf.keras.Model] = None):
        super().__init__()
        self._frontend = frontend
        self._encoder = encoder
        self._pool = tf.keras.Sequential([
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Flatten(),
        ])
        self._head = tf.keras.layers.Dense(num_outputs, activation=None)

    def call(self, inputs: tf.Tensor, training: bool = True):
        output = inputs
        if self._frontend is not None:
            output = self._frontend(output, training=training)  # pylint: disable=not-callable
            output = tf.expand_dims(output, -1)
        if self._encoder:
            output = self._encoder(output, training=training)
        output = self._pool(output)
        return self._head(output)
