import math

import tensorflow as tf


def gabor_impulse_response(t: tf.Tensor, center: tf.Tensor,
                           fwhm: tf.Tensor) -> tf.Tensor:
  """Computes the gabor impulse response."""
  denominator = 1.0 / (tf.math.sqrt(2.0 * math.pi) * fwhm)
  gaussian = tf.exp(tf.tensordot(1.0 / (2. * fwhm**2), -t**2, axes=0))
  center_frequency_complex = tf.cast(center, tf.complex64)
  t_complex = tf.cast(t, tf.complex64)
  sinusoid = tf.math.exp(
      1j * tf.tensordot(center_frequency_complex, t_complex, axes=0))
  denominator = tf.cast(denominator, dtype=tf.complex64)[:, tf.newaxis]
  gaussian = tf.cast(gaussian, dtype=tf.complex64)
  return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401) -> tf.Tensor:
  """Computes the gabor filters from its parameters for a given size.

  Args:
    kernel: tf.Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.

  Returns:
    A tf.Tensor<float>[filters, size].
  """
  return gabor_impulse_response(
      tf.range(-(size // 2), (size + 1) // 2, dtype=tf.float32),
      center=kernel[:, 0], fwhm=kernel[:, 1])

def gaussian_lowpass(sigma: tf.Tensor, filter_size: int):
  """Generates gaussian windows centered in zero, of std sigma.

  Args:
    sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.

  Returns:
    A tf.Tensor<float>[1, filter_size, C, 1].
  """
  sigma = tf.clip_by_value(
      sigma, clip_value_min=(2. / filter_size), clip_value_max=0.5)
  t = tf.range(0, filter_size, dtype=tf.float32)
  t = tf.reshape(t, (1, filter_size, 1, 1))
  numerator = t - 0.5 * (filter_size - 1)
  denominator = sigma * 0.5 * (filter_size - 1)
  return tf.math.exp(-0.5 * (numerator / denominator)**2)