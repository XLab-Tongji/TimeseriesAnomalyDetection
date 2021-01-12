# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support


class clstm_clf(object):
    """
    A Conv-LSTM classifier
    CNN+LSTM+DNN
    Reference: A C-LSTM Neural Network for Text Classification
    """
    def __init__(self, config):
        self.data_window = config.data_window
        self.feature_size = config.feature_size #the default value is 1 dimension, 60 items per window
        self.num_classes = config.num_classes
        self.filter_size = config.filter_size
        self.num_filters = config.num_filters
        self.hidden_size = self.num_filters
        self.dense_size = config.dense_size
        self.num_layers = config.num_layers #2 layers of BiLSTM
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.batch_size = 1
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_window, self.feature_size], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        #self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        #expand_dims
        inputs = tf.expand_dims(self.input_x, -1)
        #inputs [batch_size x data_window x feature_size x 1]

        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        # First convolutional layer
        with tf.variable_scope('conv1-%s' % self.filter_size,reuse=tf.AUTO_REUSE):
            # [filter size, feature size, channels, number of filters]
            filter_shape = [self.filter_size, self.feature_size, 1, self.num_filters]
            W1 = tf.get_variable('weights1', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable('biases1', [self.num_filters], initializer=tf.constant_initializer(0.0))

            # Convolution
            conv1 = tf.nn.conv2d(inputs,
                                W1,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name='conv1')
            #conv1 [batch_size x 60 x 1 x num_filters]
            # Activation function
            #h1 = tf.nn.tanh(tf.nn.bias_add(conv1, b1), name='tanh1')
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name='relu1')
        #First max pooling layer
        with tf.variable_scope('max-pooling1'):
            pooled1 = tf.nn.max_pool(
                                    h1,
                                    ksize=[1, 2, 1, 1],
                                    strides=[1, 2, 1, 1],
                                    padding='SAME',
                                    name="pool1")
        #pooled1 [batch_size x 30 x 1 x num_filters]

        # Second convolutional layer
        with tf.variable_scope('conv2-%s' % self.filter_size,reuse=tf.AUTO_REUSE):
            # [filter size, feature size, channels, number of filters]
            filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters]
            W2 = tf.get_variable('weights2', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b2 = tf.get_variable('biases2', [self.num_filters], initializer=tf.constant_initializer(0.0))

            # Convolution
            conv2 = tf.nn.conv2d(pooled1,
                                W2,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name='conv2')
            #conv2 [batch_size x 30 x 1 x num_filters]
            # Activation function
            #h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name='tanh2')
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')

        # Second max pooling layer
        with tf.variable_scope('max-pooling2'):
            pooled2 = tf.nn.max_pool(
                                    h2,
                                    ksize=[1, 2, 1, 1],
                                    strides=[1, 2, 1, 1],
                                    padding='SAME',
                                    name="pool2")

        # add the third layer
        with tf.variable_scope('conv3-%s' % self.filter_size,reuse=tf.AUTO_REUSE):
            # [filter size, feature size, channels, number of filters]
            filter_shape = [self.filter_size, self.feature_size, self.num_filters, self.num_filters]
            W3 = tf.get_variable('weights3', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b3 = tf.get_variable('biases3', [self.num_filters], initializer=tf.constant_initializer(0.0))

            # Convolution
            conv3 = tf.nn.conv2d(pooled2,
                                 W3,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv2')
            # conv2 [batch_size x 30 x 1 x num_filters]
            # Activation function
            # h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name='tanh3')
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3, name='relu3'))

            # Second max pooling layer
        with tf.variable_scope('max-pooling3'):
            pooled3 = tf.nn.max_pool(
                h3,
                ksize=[1, 2, 1, 1],
                strides=[1, 2, 1, 1],
                padding='SAME',
                name="pool3")

        #pooled2 [batch_size x 15 x 1 x num_filters]
        rnn_inputs = tf.squeeze(pooled3, [2])
        #[batch_size x 15 x num_filters]

        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to LSTM cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Feed the CNN outputs to LSTM network
        # rnn_inputs [batch_size x max_feature_length x num_filters*filter_num]
        with tf.variable_scope('LSTM',reuse=tf.AUTO_REUSE):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               rnn_inputs,
                                               initial_state=self._initial_state)
            self.final_state = state

        # Softmax output layer
        with tf.variable_scope('softmax',reuse=tf.AUTO_REUSE):
            softmax_w1 = tf.get_variable('softmax_w1', shape=[self.hidden_size, self.dense_size], dtype=tf.float32)
            softmax_b1 = tf.get_variable('softmax_b1', shape=[self.dense_size], dtype=tf.float32)
            softmax_w2 = tf.get_variable('softmax_w2', shape=[self.dense_size, self.num_classes], dtype=tf.float32)
            softmax_b2 = tf.get_variable('softmax_b2', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w1)
            self.l2_loss += tf.nn.l2_loss(softmax_b1)
            self.l2_loss += tf.nn.l2_loss(softmax_w2)
            self.l2_loss += tf.nn.l2_loss(softmax_b2)

            # logits
            #dense1 = tf.nn.tanh(tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w1) + softmax_b1, name='dense1')
            dense1 = tf.nn.relu(tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w1) + softmax_b1,
                                name='dense1')
            self.logits = tf.matmul(dense1, softmax_w2) + softmax_b2
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        ''' comment at the moment
               # Accuracy
               with tf.name_scope('accuracy'):
                   correct_predictions = tf.equal(self.predictions, self.input_y)
                   self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
                   self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
               '''

        with tf.Graph().as_default():
            prediction_integer = tf.cast(tf.round(self.predictions), tf.int64,
                                         name="Prediction_Integer")
            ## Define operations for {true, false} {positive, negative} predictions
            TP = tf.count_nonzero(prediction_integer * self.input_y,
                                  name='True_Positives', dtype=tf.int64)

            TN = tf.count_nonzero((prediction_integer - 1) * (self.input_y - 1),
                                  name="True_Negatives", dtype=tf.int64)

            FP = tf.count_nonzero(prediction_integer * (self.input_y - 1),
                                  name="False_Positives", dtype=tf.int64)

            FN = tf.count_nonzero((prediction_integer - 1) * self.input_y,
                                  name="False_Negatives", dtype=tf.int64)

            # accuracy ::= (TP + TN) / (TN + FN + TP + FP)
            self.accuracy = tf.divide(TP + TN, TN + FN + TP + FP, name="Accuracy")

            # precision ::= TP / (TP + FP)
            self.precision = tf.divide(TP, TP + FP, name="Precision")

            # recall ::= TP / (TP + FN)
            self.recall = tf.divide(TP, TP + FN, name="Recall")

            # F1 score ::= 2 * precision * recall / (precision + recall)
            self.f1score = tf.divide((2 * self.precision * self.recall), (self.precision + self.recall),
                                     name="F1_score")
