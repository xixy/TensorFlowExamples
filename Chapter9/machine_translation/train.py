#coding=utf-8
import tensorflow as tf
from data_util import *
from parameters import *
from model import *


if __name__ == '__main__':
	
	def run_epoch(model, saver, step, sess):
		'''
		运行一个epoch
		Args:
			model: NMT模型
			saver: tf.train.Saver() 用来存储模型
			step: 记录训练了多少个batch
			sess: 会话
		Return:
			step: 记录训练了多少个batch，增量式
		'''
		for src_input, src_size, trg_input, trg_label, trg_size in make_batch(en_ids_file, zh_ids_file, BATCH_SIZE):
			feed_dict = {
				model.src_input : src_input,
				model.src_size : src_size,
				model.trg_input : trg_input,
				model.trg_label : trg_label,
				model.trg_size : trg_size
			}
			cost, _ = sess.run(
				[model.cost_per_token, model.train_op],
				feed_dict = feed_dict
				)
			if step % 10 == 0:
				print 'After %d steps, per token cost is %.3f' % (step, cost)
			if step % 40 == 0:
				saver.save(sess, CHECKPOINT_PATH, global_step = step)
			step += 1
		return step

	with tf.variable_scope('nmt_model', reuse = None):
		model = Model()
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	step = 1
	for i in range(NUM_EPOCH):
		step = run_epoch(model, saver, step, sess)