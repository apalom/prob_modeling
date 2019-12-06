import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


class TF_Logistic_Classifier_NR:
    def __init__(self, D, zero_init=False, reg=1.0):
        '''
        :param D: length of feature vector
        :param zero_init: whether use all zero initialization
        :param reg: regularization strength ( precision of prior))
        '''
        self.D = D
        self.zero_init = zero_init
        self.reg = reg

        if self.zero_init:
            self.initializer = tf.initializers.zeros()
        else:
            self.initializer = tf.initializers.glorot_normal()

        self._build_graph()

    def _build_graph(self):
        # model parameters( weights)
        self.w = tf.Variable(self.initializer(shape=[self.D, 1]))

        # Design Matrix and target label
        self.X = tf.placeholder(tf.float32, shape=[None, self.D, ])
        self.t = tf.placeholder(tf.float32, shape=[None, 1])
        self.y = tf.sigmoid(self.X @ self.w)

        self.grad = tf.transpose(self.X) @ (self.y - self.t) + self.w * self.reg

        R = tf.linalg.diag(tf.reshape(self.y * (1 - self.y), shape=[-1]))
        H = tf.transpose(self.X) @ R @ self.X + tf.eye(self.D) * self.reg

        # key step
        self.update_op = tf.assign(self.w, self.w - tf.linalg.inv(H) @ self.grad)

        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, t, num_iter=10, verbose=False):
        '''
        :param X: Design Matrix, N by D
        :param t: target, N by None
        :return: self
        '''

        self.train_hist = []
        for i in range(num_iter):
            train_feed_dict = {self.X: X, self.t: t.reshape(-1, 1)}
            _, w = self.sess.run([self.update_op, self.w], feed_dict=train_feed_dict)

            if verbose:
                print('iter %3d: w = ' % (i + 1), w.reshape(-1))
        return self

    def predict(self, X):
        test_feed_dict = {self.X: X}

        y_pred = self.sess.run(self.y, feed_dict=test_feed_dict).reshape(-1)
        y_pred = (y_pred > 0.5).astype(np.int)

        return y_pred


class TF_Logistic_Classifier_SGD:
    def __init__(self, D, zero_init=False, reg=1.0, lr=0.1):
        '''
        :param D: length of feature vector
        :param zero_init: whether use all zero initialization
        :param reg: regularization strength ( precision of prior))
        '''
        self.D = D
        self.zero_init = zero_init
        self.reg = reg
        self.lr = lr

        if self.zero_init:
            self.initializer = tf.initializers.zeros()
        else:
            self.initializer = tf.initializers.glorot_normal()

        self._build_graph()

    def _build_graph(self):
        # model parameters( weights)
        self.w = tf.Variable(self.initializer(shape=[self.D, 1]))

        # Design Matrix and target label
        self.X = tf.placeholder(tf.float32, shape=[None, self.D, ])
        self.t = tf.placeholder(tf.float32, shape=[None])

        self.y = tf.reshape(tf.sigmoid(self.X @ self.w), shape=[-1])

        self.mle_loss = - tf.reduce_sum(self.t * tf.log(self.y) + (1 - self.t) * tf.log(1 - self.y))
        self.prior_loss = tf.reduce_sum(self.w * self.w * self.reg)
        self.map_loss = self.mle_loss + self.prior_loss

        # minimizer
        self.min_opt = tf.train.AdamOptimizer(self.lr)

        # minimizing step
        self.min_step = self.min_opt.minimize(self.map_loss)

        # GPU settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, t, num_iter=10, verbose=True):
        '''
        :param X: Design Matrix, N by D
        :param t: target, N by None
        :return: self
        '''

        self.train_hist = []
        for i in range(num_iter):
            train_feed_dict = {self.X: X, self.t: t}
            _, map_loss, w = self.sess.run([self.min_step, self.map_loss, self.w], feed_dict=train_feed_dict)
            if verbose:
                print("iter %3d: map loss = %f" % (i + 1, map_loss), " w = ", w.reshape(-1))

            self.train_hist.append(map_loss)
        return self

    def predict(self, X):
        test_feed_dict = {self.X: X}

        y_pred = self.sess.run(self.y, feed_dict=test_feed_dict)
        y_pred = (y_pred > 0.5).astype(np.int)

        return y_pred


if __name__ == '__main__':
    train_dat = np.genfromtxt("train.csv", delimiter=",")
    test_dat = np.genfromtxt("test.csv", delimiter=',')

    x_train = train_dat[:, :-1]
    y_train = train_dat[:, -1]

    x_test = test_dat[:, :-1]
    y_test = test_dat[:, -1]

    # attribute normalization
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    mean_train = np.average(x_train, axis=0)
    var_train = np.var(x_train, axis=0)
    x_train -= mean_train
    x_train /= np.sqrt(var_train)
    x_test -= mean_train
    x_test /= np.sqrt(var_train)



    clf = TF_Logistic_Classifier_NR( x_train.shape[1],zero_init=True, reg=1.0)
    clf.fit( x_train, y_train, 10,verbose=True)

    train_acc = accuracy_score( y_train, clf.predict( x_train))
    test_acc = accuracy_score( y_test, clf.predict( x_test))
    print("Newton Raphson, train acc = %g, test acc = %g" % ( train_acc, test_acc))



    clf = TF_Logistic_Classifier_SGD(x_train.shape[1], zero_init=True, reg=1.0, lr = 0.1)
    clf.fit(x_train, y_train, 100, verbose=True)

    train_acc = accuracy_score(y_train, clf.predict(x_train))
    test_acc = accuracy_score(y_test, clf.predict(x_test))
    print("SGD, train acc = %g, test acc = %g" % (train_acc, test_acc))



