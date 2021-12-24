import functools
from typing import Dict, Mapping

import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE


def db_to_linear(samples):
  return 10.0 ** (samples / 20.0)

def loudness_normalization(samples: tf.Tensor,
                           target_db: float = 15.0,
                           max_gain_db: float = 30.0):
  """Normalizes the loudness of the input signal."""
  std = tf.math.reduce_std(samples) + 1e-9
  gain = tf.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
  return gain * samples

def align(samples: tf.Tensor, seq_len: int = 16000):
  pad_length = tf.maximum(seq_len - tf.size(samples), 0)
  return tf.image.random_crop(tf.pad(samples, [[0, pad_length]]), [seq_len])

def preprocess(inputs: Mapping[str, tf.Tensor],
               transform_fns=(align, loudness_normalization)):
  """Sequentially applies the transformations to the waveform."""
  audio = tf.cast(inputs['audio'], tf.float32) / tf.int16.max
  for transform_fn in transform_fns:
    audio = transform_fn(audio)
  return audio, inputs['label']

def prepare(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
  """Prepares the datasets for training and evaluation."""
  result = {}
  for split in ['train', 'validation', 'eval']:
    ds = datasets[split]
    ds = ds.map(functools.partial(preprocess, transform_fns=transform_fns),
                num_parallel_calls=AUTOTUNE)
    result[split] = ds.batch(batch_size).prefetch(AUTOTUNE)

  return result