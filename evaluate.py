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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util

from parade import parade# , organise_text_dataset_for_lm_task

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 100, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("batch_size", 128,
                     "Size of a batch passed to optimizee during training")
flags.DEFINE_string("train_dataset", None,
                     "Path to file containing train dataset")
flags.DEFINE_string("valid_dataset", None,
                     "Path to file containing validation dataset")
flags.DEFINE_boolean("parade_tokens", False,
                     "If language model task is performed tokens have to be reordered in the way that"
                     " that allows tf.FixedLengthRecordReader to get batch every read")
if FLAGS.parade_tokens:
    main_parade_path, first_batch_parade_path = parade(
        FLAGS.train_dataset, FLAGS.batch_size, FLAGS.parade_tokens)


def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps

  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem, main_parade_path, first_batch_parade_path,
                                                         path=FLAGS.path)

  # Optimizer setup.
  if FLAGS.optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    grads_and_vars = optimizer.compute_gradients(cost_op)
    grads, v = zip(*grads_and_vars)
    grads, _ = tf.clip_by_global_norm(grads, 1.)
    update = optimizer.apply_gradients(zip(grads, v))
    # update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "L2L":
    if FLAGS.path is None:
      logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(**net_config)
    meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
    _, update, reset, cost_op, _ = meta_loss
  else:
    raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    total_time = 0
    total_cost = 0
    for i in xrange(FLAGS.num_epochs):
      # Training.
      time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                  num_unrolls)
      total_time += time
      total_cost += cost


    # Results.
    util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                     total_time, FLAGS.num_epochs)


if __name__ == "__main__":
  tf.app.run()
