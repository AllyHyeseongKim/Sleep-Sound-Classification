import melfilters

import numpy as np

import tensorflow as tf


class PreempInit(tf.keras.initializers.Initializer):
  """Keras initializer for the pre-emphasis.

  Returns a Tensor to initialize the pre-emphasis layer of a Leaf instance.

  Attributes:
    alpha: parameter that controls how much high frequencies are emphasized by
      the following formula output[n] = input[n] - alpha*input[n-1] with 0 <
      alpha < 1 (higher alpha boosts high frequencies)
  """

  def __init__(self, alpha=0.97):
    self.alpha = alpha

  def __call__(self, shape, dtype=None):
    assert shape == (
        2, 1, 1), 'Cannot initialize preemp layer of size {}'.format(shape)
    preemp_arr = np.zeros(shape)
    preemp_arr[0, 0, 0] = -self.alpha
    preemp_arr[1, 0, 0] = 1
    return tf.convert_to_tensor(preemp_arr, dtype=dtype)

  def get_config(self):
    return self.__dict__

class GaborInit(tf.keras.initializers.Initializer):
  """Keras initializer for the complex-valued convolution.

  Returns a Tensor to initialize the complex-valued convolution layer of a
  Leaf instance with Gabor filters designed to match the
  frequency response of standard mel-filterbanks.

  If the shape has rank 2, this is a complex convolution with filters only
  parametrized by center frequency and FWHM, so we initialize accordingly.
  In this case, we define the window len as 401 (default value), as it is not
  used for initialization.
  """

  def __init__(self, **kwargs):
    kwargs.pop('n_filters', None)
    self._kwargs = kwargs

  def __call__(self, shape, dtype=None):
    n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
    window_len = 401 if len(shape) == 2 else shape[0]
    gabor_filters = melfilters.Gabor(
        n_filters=n_filters, window_len=window_len, **self._kwargs)
    if len(shape) == 2:
      return gabor_filters.gabor_params_from_mels
    else:
      even_indices = tf.range(shape[2], delta=2)
      odd_indices = tf.range(start=1, limit=shape[2], delta=2)
      filters = gabor_filters.gabor_filters
      filters_real_and_imag = tf.dynamic_stitch(
          [even_indices, odd_indices],
          [tf.math.real(filters), tf.math.imag(filters)])
      return tf.transpose(filters_real_and_imag[:, tf.newaxis, :], [2, 1, 0])

  def get_config(self):
    return self._kwargs