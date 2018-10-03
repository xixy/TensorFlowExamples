#coding=utf-8

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
	# (batch, INPUT_NODE)
	x = tf.placeholder(
		name = 'x-input',
		shape = [None, mnist_inference.INPUT_NODE],
		dtype = tf.float32
		)
	# (batch, OUTPUT_NODE)
	y_ = tf.placeholder(
		name = 'y-input',
		shape = [None, mnist_inference.OUTPUT_NODE],
		dtype = tf.float32
		)
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	# (batch, OUTPUT_NODE)
	y = mnist_inference.inference(x, regularizer)
	# 定义训练step
	global_step = tf.Variable(0, 
		trainable = False)

	# 加入滑动平均模型
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY,
		global_step
		)
	# 针对所有的可训练变量进行滑动平均
	variables_averages_op = variable_averages.apply(
		tf.trainable_variables()
		)

	# (batch,)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits = y,
		labels = tf.argmax(y_, 1)
		)
	# 计算平均损失
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	learning_rate = tf.train.exponential_decay(
		learning_rate = LEARNING_RATE_BASE,
		global_step = global_step,
		decay_steps = mnist.train.num_examples / BATCH_SIZE,
		decay_rate = LEARNING_RATE_DECAY
		)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
		loss,
		global_step = global_step
		)

	with tf.control_dependencies([train_step, variables_averages_op]):
		train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run(
				[train_op, loss, global_step],
				feed_dict = {x: xs, y_: ys}
				)
			if i % 1000 == 0:
				print 'After %d training steps, loss on training batch is %g.' % (step, loss_value)
				saver.save(
					sess = sess, 
					save_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
					global_step = global_step)

def main():
	mnist = input_data.read_data_sets('./data/', one_hot = True)
	train(mnist)
if __name__ == '__main__':
	main()

