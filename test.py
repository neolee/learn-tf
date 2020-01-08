from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print('TensorFlow version:', tf.__version__)
if tf.test.is_gpu_available():
    print('GPU available')
print()

cpus = tf.config.experimental.list_physical_devices('CPU')
print('Number of available CPU:', len(cpus))

gpus = tf.config.experimental.list_physical_devices('GPU')
print('Number of available GPU:', len(gpus))
print()

tf.debugging.set_log_device_placement(True)
print('Device placement log: enabled')
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
