#coding=utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import numpy as np
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		# shape = (batch, length, width, channels)
		x = tf.placeholder(
			name = 'x-input',
			shape = [None, 
			mnist_inference.IMAGE_SIZE, 
			mnist_inference.IMAGE_SIZE, 
			mnist_inference.NUM_CHANNELS],
			dtype = tf.float32
			)
		# (batch, OUTPUT_NODE)
		y_ = tf.placeholder(
			name = 'y-input',
			shape = [None, mnist_inference.OUTPUT_NODE],
			dtype = tf.float32
			)
		images = mnist.validation.images
		shape = images.shape
		images = np.reshape(mnist.validation.images,
			[-1, 
			mnist_inference.IMAGE_SIZE, 
			mnist_inference.IMAGE_SIZE,
			mnist_inference.NUM_CHANNELS]
			)
		validate_feed = {
		x : images,
		y_ : mnist.validation.labels
		}
		y = mnist_inference.inference(x, False, None)
		correct_prediction = tf.equal(
			tf.argmax(y, 1),
			tf.argmax(y_, 1)
		)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		variable_averages = tf.train.ExponentialMovingAverage(
			mnist_train.MOVING_AVERAGE_DECAY
			)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		while True:
			with tf.Session() as sess:
				# 通过checkpoint找到目录下最新模型的文件名
				ckpt = tf.train.get_checkpoint_state(
					mnist_train.MODEL_SAVE_PATH
					)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
					print 'After %s train steps, validation accuracy = %g' % (global_step, accuracy_score)
				else:
					print 'No checkpoint file found'
					return

			time.sleep(EVAL_INTERVAL_SECS)

def main():
	mnist = input_data.read_data_sets('./data/', one_hot = True)
	evaluate(mnist)
if __name__ == '__main__':
	main()