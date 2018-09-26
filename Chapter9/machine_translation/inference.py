#coding=utf-8
import tensorflow as tf
from data_util import *
from parameters import *
from model import *

def inference(sess, model, sentence):
	output_op = model.inference(sentence)
	return sess.run(output_op)

if __name__ == '__main__':
	with tf.variable_scope('nmt_model', reuse = None):
		model = Model()
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, CHECKPOINT_RESTORE_PATH)
	test_sentence = [10, 20, 30, 40, 50]
	print inference(sess, model, test_sentence)


