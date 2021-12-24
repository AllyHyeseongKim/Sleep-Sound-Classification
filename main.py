from data import load
from models import AudioClassifier
from frontend import Leaf
from preprocess import prepare
from tflite_coverter import export_tflite

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)


data_path = '{path}/sleep scoring/'

if __name__ == "__main__":
    learning_rate = 1e-4
    metric = 'sparse_categorical_accuracy'
    num_epochs = 10
    batch_size = 256

    sleep_scoring_path = data_path
    dataset, classes = load(path=sleep_scoring_path)
    dataset = prepare(dataset, batch_size=batch_size)
    print("After Preprocessing: ")
    print(dataset)

    frontend = Leaf()
    model = AudioClassifier(num_outputs=len(classes), frontend=frontend)
    #print(model.summary())
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[metric]
    )
    ckpt_path = './temp/checkpoint'
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor=f'val_{metric}',
        mode='max',
        save_best_only=True
    )
    history = model.fit(
        dataset['train'],
        validation_data=dataset['validation'],
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[callback]
    )

    loss, accuracy = model.evaluate(dataset['eval'])
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    saved_model_path = './results/sleep_sound_model/10-256'
    model.save(saved_model_path)
    export_tflite(saved_model_path)
