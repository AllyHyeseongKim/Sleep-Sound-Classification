import os

import tensorflow as tf


def export_tflite(saved_model_dir):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  converter._experimental_lower_tensor_list_ops = False

  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  #tflite_quant_model = converter.convert()
  #open("converted_model.tflite", "wb").write(tflite_quant_model)
  tflite_model = converter.convert()

  converted_model_path = os.path.join(saved_model_dir, 'converted_model.tflite')
  open(converted_model_path, "wb").write(tflite_model)