# -*- coding=utf-8 -*-
from datetime import datetime
import math
import time
import tensorflow as tf

# total 100 batch
batch_size = 32
num_batches = 100


# show the tensor for every input
def print_activations(t):
    print t.op.name, '', t.get_shape().as_list()


# weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        # Varibale 每次都会创建一个新的变量
        # kernel 参数，前两个是卷积核的大小，第三个是当前的深度，第三个是卷积核的深度
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv1)

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], \
                           padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        # Varibale 每次都会创建一个新的变量
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], \
                           padding='VALID', name='pool1')
    print_activations(pool2)

    with tf.name_scope('conv2') as scope:
        # Varibale 每次都会创建一个新的变量
        kernel = tf.Variable(tf.truncated_normal([5, 5, 192, 384],
                                                 dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        # Varibale 每次都会创建一个新的变量
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        # Varibale 每次都会创建一个新的变量
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')
    print_activations(pool5)

    return pool5, parameters


# w_fc1 = weight_variable([256, 4096])
# b_fc1 = bias_variable([4096])
# fc1 = tf.nn.relu(tf.matmul(pool5, w_fc1) + b_fc1)
#
# w_fc2 = weight_variable([4096, 4096])
# b_fc2 = bias_variable([4096])
# fc2 = tf.nn.relu(tf.matmul(fc, w_fc2) + b_fc2)
#
# w_fc3 = weight_variable([4096, 1000])
# b_fc3 = bias_variable([1000])
# fc3 = tf.nn.relu(tf.matmul(fc2, w_fc3) + b_fc3)


def time_tensoflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print '%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration)
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print '%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd)


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            time_tensoflow_run(sess, pool5, "Forward")
            objective = tf.nn.l2_loss(pool5)
            grad = tf.gradients(objective, parameters)
            time_tensoflow_run(sess, grad, "Forward-backward")


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    run_benchmark()