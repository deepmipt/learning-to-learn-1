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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

from some_useful_functions import create_path, add_postfix_to_path_string
from parade import parade #, organise_text_dataset_for_lm_task

import meta
import util

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 100, "Log period.")
flags.DEFINE_integer("evaluation_period", 1000, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("batch_size", 128,
                     "Size of a batch passed to optimizee during training")
flags.DEFINE_string("train_dataset", None,
                     "Path to file containing train dataset")
flags.DEFINE_string("valid_dataset", None,
                     "Path to file containing validation dataset")
flags.DEFINE_boolean("parade_tokens", False,
                     "If language model task is performed tokens have to be reordered in the way that"
                     " allows tf.FixedLengthRecordReader to get batch every read")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")
#
# if FLAGS.parade_tokens:
#   if FLAGS.train_dataset is not None:
#     main_parade_path = add_postfix_to_path_string(FLAGS.train_dataset, '_parade')
#     first_batch_parade_path = add_postfix_to_path_string(FLAGS.train_dataset, '_parade_first_batch')
#     if not os.path.exists(main_parade_path) or not os.path.exists(first_batch_parade_path):
#       [main_parade_path, first_batch_parade_path] = parade(FLAGS.train_dataset, FLAGS.batch_size, True)
# else:
#   main_parade_path = None
#   first_batch_parade_path = None
if FLAGS.parade_tokens == 'False':
    parade_tokens = False
else:
    parade_tokens = FLAGS.parade_tokens
if parade_tokens:
    main_parade_path, first_batch_parade_path = parade(
        FLAGS.train_dataset, FLAGS.batch_size, True)
else:
    main_parade_path = None
    first_batch_parade_path = None

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  # if FLAGS.save_path is not None:
  #   if os.path.exists(FLAGS.save_path):
  #     raise ValueError("Folder {} already exists".format(FLAGS.save_path))
  #   else:
  #     os.mkdir(FLAGS.save_path)

  # Problem.
  problem, net_config, net_assignments = util.get_config(
      FLAGS.problem, main_parade_path, first_batch_parade_path)

  # Optimizer setup.
  optimizer = meta.MetaOptimizer(**net_config)
  minimize = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)
  step, update, reset, cost_op, _ = minimize

  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()
    writer = tf.summary.FileWriter('summary')
    writer.add_graph(tf.get_default_graph())
    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls)
      total_time += time
      total_cost += cost

      # Logging.
      if (e + 1) % FLAGS.log_period == 0:
        util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         FLAGS.log_period)
        total_time = 0
        total_cost = 0

      # Evaluation.
      if (e + 1) % FLAGS.evaluation_period == 0:
        eval_cost = 0
        eval_time = 0
        for _ in xrange(FLAGS.evaluation_epochs):
          time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                      num_unrolls)
          eval_time += time
          eval_cost += cost

        util.print_stats("EVALUATION", eval_cost, eval_time,
                         FLAGS.evaluation_epochs)

        if FLAGS.save_path is not None and eval_cost < best_evaluation:
          print("Removing previously saved meta-optimizer")
          for f in os.listdir(FLAGS.save_path):
            os.remove(os.path.join(FLAGS.save_path, f))
          print("Saving meta-optimizer to {}".format(FLAGS.save_path))
          optimizer.save(sess, FLAGS.save_path)
          best_evaluation = eval_cost


if __name__ == "__main__":
  tf.app.run()
