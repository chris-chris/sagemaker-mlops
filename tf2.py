"""Tensorflow2 Keras Version Model Training"""

import argparse
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

def model(train_dataset, test_dataset, args):
  """Generate a simple model"""
  model = Sequential(
      [
        Dense(256, input_shape=(8,)),
        Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
      ])
  # eval_callback = LambdaCallback(
  #     on_epoch_end=lambda epoch, logs: logs.update(
  #         {'mean_logits': K.eval(mean)}
  #     ))
  model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='binary_crossentropy')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)

  model.fit(
      train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE),
      validation_data=test_dataset.batch(BATCH_SIZE),
      epochs=100, verbose=0, callbacks=[early_stopping])

  return model


def _load_data(base_dir):
  """Load device failure data"""

  total_data = np.load(os.path.join(base_dir, 'total_data.npy'))
  total_label = np.load(os.path.join(base_dir, 'total_label.npy'))
  test_split = int(len(total_data)*0.2)

  train_data = total_data[:-test_split]   # 8:2
  train_label = total_label[:-test_split] # 8:2
  test_data = total_data[-test_split:]   # 8:2
  test_label = total_label[-test_split:] # 8:2

  train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))

  return train_dataset, test_dataset


def _parse_args():
  parser = argparse.ArgumentParser()

  # Data, model, and output directories
  # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
  parser.add_argument('--model_dir', type=str)
  parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
  parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
  parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
  parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
  parser.add_argument('--learning-rate', type=float, default=0.001)

  return parser.parse_known_args()


if __name__ == "__main__":
  args, unknown = _parse_args()

  print("ars: ", args)
  print(f"learning_rate: {args.learning_rate}")

  train_dataset, test_dataset = _load_data(args.train)

  device_failure_model = model(train_dataset, test_dataset, args)

  loss = device_failure_model.evaluate(test_dataset.batch(BATCH_SIZE))
  print(f"test_loss: {loss}")
  tf.summary.scalar("test_loss", loss)

  if args.current_host == args.hosts[0]:
    # save model to an S3 directory with version number '00000001'
    device_failure_model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')