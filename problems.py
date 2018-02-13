# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import numpy as np

from tensorflow.python.ops.init_ops import Initializer

from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset

NUMBER_OF_CHARACTERS = 256


_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def simple():
  """Simple problem: f(x) = x^2."""

  def build():
    """Builds loss graph."""
    x = tf.get_variable(
        "x",
        shape=[],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    return tf.square(x, name="x_squared")

  return build


def simple_multi_optimizer(num_dims=2):
  """Multidimensional simple problem."""

  def get_coordinate(i):
    return tf.get_variable("x_{}".format(i),
                           shape=[],
                           dtype=tf.float32,
                           initializer=tf.ones_initializer())

  def build():
    coordinates = [get_coordinate(i) for i in xrange(num_dims)]
    x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
    return tf.reduce_sum(tf.square(x, name="x_squared"))

  return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

  return build


def ensemble(problems, weights=None):
  """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
  if weights and len(weights) != len(problems):
    raise ValueError("len(weights) != len(problems)")

  build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
               for p in problems]

  def build():
    loss = 0
    for i, build_fn in enumerate(build_fns):
      with tf.variable_scope("problem_{}".format(i)):
        loss_p = build_fn()
        if weights:
          loss_p *= weights[i]
        loss += loss_p
    return loss

  return build


def _xent_loss(output, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
  return tf.reduce_mean(loss)


def mnist(layers,  # pylint: disable=invalid-name
          activation="sigmoid",
          batch_size=128,
          mode="train"):
  """Mnist classification with a multi-layer perceptron."""

  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

  # Network.
  mlp = snt.nets.MLP(list(layers) + [10],
                     activation=activation_op,
                     initializers=_nn_initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"


def _maybe_download_cifar10(path):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(path):
    os.makedirs(path)
  filepath = os.path.join(path, CIFAR10_FILE)
  if not os.path.exists(filepath):
    print("Downloading CIFAR10 dataset to {}".format(filepath))
    url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded {} bytes".format(statinfo.st_size))
    tarfile.open(filepath, "r:gz").extractall(path)


def cifar10(path,  # pylint: disable=invalid-name
            conv_channels=None,
            linear_layers=None,
            batch_norm=True,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
  """Cifar10 classification with a convolutional network."""

  # Data.
  _maybe_download_cifar10(path)

  # Read images and labels from disk.
  if mode == "train":
    filenames = [os.path.join(path,
                              CIFAR10_FOLDER,
                              "data_batch_{}.bin".format(i))
                 for i in xrange(1, 6)]
  elif mode == "test":
    filenames = [os.path.join(path, "test_batch.bin")]
  else:
    raise ValueError("Mode {} not recognised".format(mode))

  depth = 3
  height = 32
  width = 32
  label_bytes = 1
  image_bytes = depth * height * width
  record_bytes = label_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, record = reader.read(tf.train.string_input_producer(filenames))
  record_bytes = tf.decode_raw(record, tf.uint8)

  label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
  image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
  # height x width x depth.
  image = tf.transpose(image, [1, 2, 0])
  image = tf.div(image, 255)

  queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples,
                                dtypes=[tf.float32, tf.int32],
                                shapes=[image.get_shape(), label.get_shape()])
  enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

  # Network.
  def _conv_activation(x):  # pylint: disable=invalid-name
    return tf.nn.max_pool(tf.nn.relu(x),
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME")

  conv = snt.nets.ConvNet2D(output_channels=conv_channels,
                            kernel_shapes=[5],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=_conv_activation,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=batch_norm)

  if batch_norm:
    linear_activation = lambda x: tf.nn.relu(snt.BatchNorm()(x))
  else:
    linear_activation = tf.nn.relu

  mlp = snt.nets.MLP(list(linear_layers) + [10],
                     activation=linear_activation,
                     initializers=_nn_initializers)
  network = snt.Sequential([conv, snt.BatchFlatten(), mlp])

  def build():
    image_batch, label_batch = queue.dequeue_many(batch_size)
    label_batch = tf.reshape(label_batch, [batch_size])

    output = network(image_batch)
    return _xent_loss(output, label_batch)

  return build


def lstm_cell(inp, hidden, matrix, bias):
    hidden_size = hidden[0].get_shape().as_list()[1]
    x = tf.concat([inp, hidden[0]], 1)
    conc = tf.matmul(x, matrix) + bias
    tanh_arg, sigm_arg = tf.split(conc, [hidden_size, 3*hidden_size], axis=1)
    j = tf.tanh(tanh_arg)
    i_gate, f_gate, o_gate = tf.split(tf.sigmoid(sigm_arg), 3, axis=1)
    new_c = hidden[1] * f_gate + j * i_gate
    new_h = tf.tanh(new_c) * o_gate
    return new_h, [new_h, new_c]


def unrolled_lstm(inputs, matrix, bias, hidden):
    inps = tf.unstack(inputs)
    outputs = list()
    for inp in inps:
        output, hidden = lstm_cell(inp, hidden, matrix, bias)
        outputs.append(output)
    return tf.stack(outputs), hidden


def lstm_lm(dataset_name,
            first_batch_dataset_name,
            hidden_size=1024,
            optimizee_unrollings=10,
          batch_size=128,
          mode="train"):
  """Mnist classification with a multi-layer perceptron."""
  with open(first_batch_dataset_name, 'r') as f:
      first_batch_text = f.read()

  first_batch = tf.reshape(
      tf.cast(
          tf.decode_raw(
              tf.constant(first_batch_text),
              tf.uint8),
          tf.float32),
      [1, len(first_batch_text)])
  embeddings = tf.constant(
      [[1. if i == j else 0. for i in range(NUMBER_OF_CHARACTERS)] for j in range(NUMBER_OF_CHARACTERS)])
  # embeddings = tf.Print(embeddings, [embeddings], summarize=66000)
  reader = tf.FixedLengthRecordReader(record_bytes=(optimizee_unrollings + 1) * batch_size)
  saved_batch = tf.get_variable(
      "saved_batch",
      initializer=first_batch,
      trainable=False)
  _, record = reader.read(tf.train.string_input_producer([dataset_name]))
  # f = open(dataset_name, 'r')
  # print('(lstm_lm)f:', f)
  # print('(lstm_lm)record.shape:', record.get_shape().as_list())
  decoded = tf.cast(
      tf.reshape(
          tf.decode_raw(
              record, tf.uint8),
          [(optimizee_unrollings + 1), batch_size]),
      tf.int32)
  # print('(lstm_lm)decoded.shape:', decoded.get_shape().as_list())
  q = tf.FIFOQueue(10, [tf.int32])
  enq = q.enqueue(decoded)
  tf.train.add_queue_runner(tf.train.QueueRunner(q, [enq]))
  # bias = tf.Print(bias, [bias], message='bias: ', summarize=300)

  def build():
    lstm_matrix = tf.get_variable(
      'lstm_matrix',
      shape=[NUMBER_OF_CHARACTERS + hidden_size, 4 * hidden_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(NUMBER_OF_CHARACTERS + hidden_size)))
    bias = tf.get_variable(
      'lstm_bias',
      shape=[4 * hidden_size],
      dtype=tf.float32,
      initializer=tf.zeros_initializer()
    )
    output_matrix = tf.get_variable(
      'output_matrix',
      shape=[hidden_size, NUMBER_OF_CHARACTERS],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(hidden_size)))
    saved_state = list()
    for i in range(2):
      saved_state.append(
          tf.get_variable(
              'h%s' % i,
              shape=[batch_size, hidden_size],
              dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False))
    dec = tf.reshape(q.dequeue(), [(optimizee_unrollings + 1), batch_size])
    # dec = tf.Print(dec, [dec])
    # print('(lstm_lm)dec.shape:', dec.get_shape().as_list())
    main, last = tf.split(dec, [optimizee_unrollings, 1])
    # print('(lstm_lm)main.shape:', main.get_shape().as_list())
    # print('(lstm_lm)saved_batch.shape:', saved_batch.get_shape().as_list())
    inputs = tf.concat([tf.cast(saved_batch, tf.int32), main], 0)
    # print('(lstm_lm after concat)inputs.shape:', inputs.get_shape().as_list())
    labels = dec

    # check_inputs = tf.reshape(tf.slice(inputs, [0, 0], [(optimizee_unrollings + 1), 1]), [-1])
    # check_labels = tf.reshape(tf.slice(labels, [0, 0], [(optimizee_unrollings + 1), 1]), [-1])
    # inputs = tf.Print(inputs, [check_inputs], message='check_inputs: ', summarize=1000)
    # labels = tf.Print(labels, [check_labels], message='check_labels: ', summarize=1000)

    inputs = tf.nn.embedding_lookup(embeddings, inputs)
    # print('(lstm_lm)inputs.shape:', inputs.get_shape().as_list())
    outputs, final_state = unrolled_lstm(inputs, lstm_matrix, bias, saved_state)
    outputs = tf.reshape(outputs, [-1, hidden_size])
    logits = tf.matmul(outputs, output_matrix)


    # lstm = snt.LSTM(NUMBER_OF_CHARACTERS)
    # initial_state = lstm.initial_state(batch_size)
    # saved_state = tf.get_variable(
    #     "saved_state",
    #     initializer=initial_state,
    #     trainable=False)
    # outputs, final_state = tf.nn.dynamic_rnn(
    #   lstm, inputs, initial_state=initial_state, time_major=True)
    # print('(lstm_lm)outputs.shape:', outputs.get_shape().as_list())
    # print('(lstm_lm)final_state.shape:', final_state.get_shape().as_list())
    # Network.
    save_ops = list()
    save_ops.append(tf.assign(saved_batch, tf.cast(last, tf.float32)))
    for saved, final in zip(saved_state, final_state):
        save_ops.append(tf.assign(saved, final))
    labels = tf.reshape(labels, [-1])
    # print('(lstm_lm)outputs.shape:', outputs.get_shape().as_list())
    # print('(lstm_lm)labels.shape:', labels.get_shape().as_list())
    with tf.control_dependencies(save_ops):
      loss = _xent_loss(
          logits,
          labels)
    return loss

  return build


def splitted_lstm_lm(dataset_name,
            first_batch_dataset_name,
            hidden_size=320,
            optimizee_unrollings=10,
          batch_size=128,
          mode="train"):
  """Mnist classification with a multi-layer perceptron."""
  with open(first_batch_dataset_name, 'r') as f:
      first_batch_text = f.read()

  first_batch = tf.reshape(
      tf.cast(
          tf.decode_raw(
              tf.constant(first_batch_text),
              tf.uint8),
          tf.float32),
      [1, len(first_batch_text)])
  embeddings = tf.constant(
      [[1. if i == j else 0. for i in range(NUMBER_OF_CHARACTERS)] for j in range(NUMBER_OF_CHARACTERS)])
  # embeddings = tf.Print(embeddings, [embeddings], summarize=66000)
  reader = tf.FixedLengthRecordReader(record_bytes=(optimizee_unrollings + 1) * batch_size)
  saved_batch = tf.get_variable(
      "saved_batch",
      initializer=first_batch,
      trainable=False)
  _, record = reader.read(tf.train.string_input_producer([dataset_name]))
  # f = open(dataset_name, 'r')
  # print('(lstm_lm)f:', f)
  # print('(lstm_lm)record.shape:', record.get_shape().as_list())
  decoded = tf.cast(
      tf.reshape(
          tf.decode_raw(
              record, tf.uint8),
          [(optimizee_unrollings + 1), batch_size]),
      tf.int32)
  # print('(lstm_lm)decoded.shape:', decoded.get_shape().as_list())
  q = tf.FIFOQueue(10, [tf.int32])
  enq = q.enqueue(decoded)
  tf.train.add_queue_runner(tf.train.QueueRunner(q, [enq]))
  # bias = tf.Print(bias, [bias], message='bias: ', summarize=300)

  def build():
    i_gate_matrix = tf.get_variable(
      'i_gate_matrix',
      shape=[NUMBER_OF_CHARACTERS + hidden_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(NUMBER_OF_CHARACTERS + hidden_size)))
    f_gate_matrix = tf.get_variable(
      'f_gate_matrix',
      shape=[NUMBER_OF_CHARACTERS + hidden_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(NUMBER_OF_CHARACTERS + hidden_size)))
    o_gate_matrix = tf.get_variable(
      'o_gate_matrix',
      shape=[NUMBER_OF_CHARACTERS + hidden_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(NUMBER_OF_CHARACTERS + hidden_size)))
    j_matrix = tf.get_variable(
      'j_matrix',
      shape=[NUMBER_OF_CHARACTERS + hidden_size, hidden_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(NUMBER_OF_CHARACTERS + hidden_size)))
    lstm_matrix = tf.concat([j_matrix, i_gate_matrix, f_gate_matrix, o_gate_matrix], 1)

    i_gate_bias = tf.get_variable(
      'i_gate_bias',
      shape=[hidden_size],
      dtype=tf.float32,
      initializer=tf.zeros_initializer()
    )
    f_gate_bias = tf.get_variable(
      'f_gate_bias',
      shape=[hidden_size],
      dtype=tf.float32,
      initializer=tf.zeros_initializer()
    )
    o_gate_bias = tf.get_variable(
      'o_gate_bias',
      shape=[hidden_size],
      dtype=tf.float32,
      initializer=tf.zeros_initializer()
    )
    j_bias = tf.get_variable(
      'j_bias',
      shape=[hidden_size],
      dtype=tf.float32,
      initializer=tf.zeros_initializer()
    )
    bias = tf.concat([j_bias, i_gate_bias, f_gate_bias, o_gate_bias], 0)

    output_matrix = tf.get_variable(
      'output_matrix',
      shape=[hidden_size, NUMBER_OF_CHARACTERS],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer(stddev=4. / np.sqrt(hidden_size)))
    saved_state = list()
    for i in range(2):
      saved_state.append(
          tf.get_variable(
              'h%s' % i,
              shape=[batch_size, hidden_size],
              dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False))
    dec = tf.reshape(q.dequeue(), [(optimizee_unrollings + 1), batch_size])
    # dec = tf.Print(dec, [dec])
    # print('(lstm_lm)dec.shape:', dec.get_shape().as_list())
    main, last = tf.split(dec, [optimizee_unrollings, 1])
    # print('(lstm_lm)main.shape:', main.get_shape().as_list())
    # print('(lstm_lm)saved_batch.shape:', saved_batch.get_shape().as_list())
    inputs = tf.concat([tf.cast(saved_batch, tf.int32), main], 0)
    # print('(lstm_lm after concat)inputs.shape:', inputs.get_shape().as_list())
    labels = dec

    # check_inputs = tf.reshape(tf.slice(inputs, [0, 0], [(optimizee_unrollings + 1), 1]), [-1])
    # check_labels = tf.reshape(tf.slice(labels, [0, 0], [(optimizee_unrollings + 1), 1]), [-1])
    # inputs = tf.Print(inputs, [check_inputs], message='check_inputs: ', summarize=1000)
    # labels = tf.Print(labels, [check_labels], message='check_labels: ', summarize=1000)

    inputs = tf.nn.embedding_lookup(embeddings, inputs)
    # print('(lstm_lm)inputs.shape:', inputs.get_shape().as_list())
    outputs, final_state = unrolled_lstm(inputs, lstm_matrix, bias, saved_state)
    outputs = tf.reshape(outputs, [-1, hidden_size])
    logits = tf.matmul(outputs, output_matrix)


    # lstm = snt.LSTM(NUMBER_OF_CHARACTERS)
    # initial_state = lstm.initial_state(batch_size)
    # saved_state = tf.get_variable(
    #     "saved_state",
    #     initializer=initial_state,
    #     trainable=False)
    # outputs, final_state = tf.nn.dynamic_rnn(
    #   lstm, inputs, initial_state=initial_state, time_major=True)
    # print('(lstm_lm)outputs.shape:', outputs.get_shape().as_list())
    # print('(lstm_lm)final_state.shape:', final_state.get_shape().as_list())
    # Network.
    save_ops = list()
    save_ops.append(tf.assign(saved_batch, tf.cast(last, tf.float32)))
    for saved, final in zip(saved_state, final_state):
        save_ops.append(tf.assign(saved, final))
    labels = tf.reshape(labels, [-1])
    # print('(lstm_lm)outputs.shape:', outputs.get_shape().as_list())
    # print('(lstm_lm)labels.shape:', labels.get_shape().as_list())
    with tf.control_dependencies(save_ops):
      loss = _xent_loss(
          logits,
          labels)
    return loss

  return build