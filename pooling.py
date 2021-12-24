import impulse_responses

import tensorflow as tf


class GaussianLowpass(tf.keras.layers.Layer):
  """Depthwise pooling (each input filter has its own pooling filter).

  Pooling filters are parametrized as zero-mean Gaussians, with learnable
  std. They can be initialized with tf.keras.initializers.Constant(0.4)
  to approximate a Hanning window.
  We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
  """

  def __init__(
      self,
      kernel_size,
      strides=1,
      padding='same',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      trainable=False,
  ):

    super().__init__(name='learnable_pooling')
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.trainable = trainable

  def build(self, input_shape):
    self.kernel = self.add_weight(
        name='kernel',
        shape=(1, 1, input_shape[2], 1),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        trainable=self.trainable)

  def call(self, inputs):
    kernel = impulse_responses.gaussian_lowpass(self.kernel, self.kernel_size)
    outputs = tf.expand_dims(inputs, axis=1)
    outputs = tf.nn.depthwise_conv2d(
        outputs,
        kernel,
        strides=(1, self.strides, self.strides, 1),
        padding=self.padding.upper())
    return tf.squeeze(outputs, axis=1)