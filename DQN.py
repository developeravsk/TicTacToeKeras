import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

input_size = 9
output_size = 9


class DQN:
    def __init__(self, sess, learning_rate=1e-2, name="main"):

        self.session = sess
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network(learning_rate)

    def _build_network(self, learning_rate=1e-2):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, shape=[None, self.input_size], name="input_x")
            self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            l1 = tf.layers.dense(inputs=self._X, units=150, activation=tf.nn.relu)
            l2 = tf.layers.dense(inputs=l1, units=120, activation=tf.nn.relu)
            l3 = tf.layers.dense(inputs=l2, units=100, activation=tf.nn.relu)
            l4 = tf.layers.dense(inputs=l3, units=100, activation=tf.nn.relu)
            l5 = tf.layers.dense(inputs=l4, units=80, activation=tf.nn.relu)
            l6 = tf.layers.dense(inputs=l5, units=80, activation=tf.nn.relu)

            self._Qpred = tf.layers.dense(inputs=l6, units=self.output_size)
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
            self._train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)
            self.saver = tf.train.Saver()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, state, values):
        return self.session.run(self._train, feed_dict={self._X: state, self._Y: values})

    def save(self):
        save_path = self.saver.save(self.session, "./data/save.ckpt")
        print("Model saved in file: %s" % save_path)

    def best_save(self):
        save_path = self.saver.save(self.session, "./data/save_best.ckpt")
        print("Best model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.session, "./data/save.ckpt")
        print("Model restored!")