"""
ref: https://towardsdatascience.com/custom-loss-function-in-tensorflow-eebcd7fed17a


https://keras.io/examples/vision/supervised-contrastive-learning/
"""

import tensorflow as tf

def create_huber(threshold=1.0): 
  def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < threshold
    squared_loss = tf.square(error) / 2
    linear_loss = threshold * tf.abs(error) - threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)
  return huber_fn

class CustomAccuracy(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred-y_true))
    rmse = tf.math.sqrt(mse)
    return rmse / tf.reduce_mean(tf.square(y_true)) - 1
 
model.compile(optimizer=Adam(learning_rate=0.001), loss=CustomAccuracy(), metrics=['mae', 'mse'])
