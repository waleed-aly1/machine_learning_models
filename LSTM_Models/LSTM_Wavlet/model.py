import tensorflow as tf
import numpy as np

timesteps = 2050
n_classes = 1
n_unitsfc = 32
residuals = False
decay = False
batch_size = 64

class MLP:
    def __init__(self, n_features, n_classes=n_classes):
        # tf.reset_default_graph()
        self.sess = tf.Session()
        self.X = tf.placeholder(tf.float32, [None, n_features], 'X')
        self.Y = tf.placeholder(tf.float32, [None, n_classes], 'Y')

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001#0.003
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   100000, 0.97, staircase=True)

        d = tf.layers.dense(self.X, n_unitsfc, tf.nn.relu)#, kernel_regularizer='l2', bias_regularizer='l2')

        self.out = tf.layers.dense(d, n_classes)

        self.loss = tf.reduce_mean(tf.squared_difference(self.out,self.Y))
        #self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.Y, self.logits))

        if decay:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, name='Saved_models/1'):
        # rest_vars = [v for v in tf.global_variables() if 'ae' not in v.name]
        save_path = tf.train.Saver().save(self.sess, name)

    def load_model(self, name='Saved_models/'):
        tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint(name + './'))

    def fit(self, x, y, epochs=0):
        total_batch = int(len(x) / batch_size)
        mod = len(x) % batch_size
        for epoch in range(epochs):
            avg_cost = 0
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x[i * batch_size: i * batch_size + batch_size]
                batch_y = y[i * batch_size: i * batch_size + batch_size]
                _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})#self.layer_3,self.unpool_1,
                avg_cost += c / total_batch

            # Remaining
            batch_x = x[total_batch * batch_size: total_batch * batch_size + mod]
            batch_y = y[total_batch * batch_size: total_batch * batch_size + mod]
            _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})#self.layer_3,self.unpool_1,
            if total_batch > 5:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))#, "Time:", time.time() - start, "s")

    def predict(self, x_ts):
        preds = self.sess.run(self.out, {self.X: x_ts})
        return preds

    def predict_batch(self, x_ts):
        total_batch = int(len(x_ts) / batch_size)
        mod = len(x_ts) % batch_size
        preds = np.empty(len(x_ts))
        for i in range(total_batch):
            batch_x = x_ts[i * batch_size: i * batch_size + batch_size]
            preds[i * batch_size: i * batch_size + batch_size] = self.sess.run(self.out, {self.X: batch_x})  # self.layer_3,self.unpool_1,
        # Remaining
        batch_x = x_ts[total_batch * batch_size: total_batch * batch_size + mod]
        preds[total_batch * batch_size: total_batch * batch_size + mod] = self.sess.run(self.out, {self.X: batch_x})
        return preds
