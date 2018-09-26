#coding=utf-8
import tensorflow as tf
from data_util import *
from parameters import *

class Model(object):
	"""docstring for Model"""
	def __init__(self):
		'''
		'''
		# 定义输入
		self.src_input = tf.placeholder(
			shape = [None, None], 
			name = 'src_input', 
			dtype = tf.int32
			)
		self.src_size = tf.placeholder(
			shape = [None],
			name = 'src_size',
			dtype = tf.int32
			)
		self.trg_input = tf.placeholder(
			shape = [None, None], 
			name = 'trg_input', 
			dtype = tf.int32			
			)
		self.trg_size = tf.placeholder(
			shape = [None], 
			name = 'trg_size', 
			dtype = tf.int32			
			)
		# (batch_size , max_word_length)
		self.trg_label = tf.placeholder(
			shape = [None, None],
			name = 'trg_label',
			dtype = tf.int32
			)
		# 定义模型
		self.encode_cell = tf.nn.rnn_cell.MultiRNNCell(
			[tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
			)
		# 定义模型
		self.decode_cell = tf.nn.rnn_cell.MultiRNNCell(
			[tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
			)

		self.src_embedding = tf.get_variable(
			name = 'src_embedding',
			shape = [en_vocab_size, HIDDEN_SIZE]
			)
		self.trg_embedding = tf.get_variable(
			name = 'trg_embedding',
			shape = [zh_vocab_size, HIDDEN_SIZE]
			)		
		# 定义softmax层变量
		if SHARED_EMD_AND_SOFTMAX:
			self.softmax_weight = tf.transpose(self.trg_embedding)
		else:
			self.softmax_weight = tf.get_variable(
				shape = [HIDDEN_SIZE, zh_vocab_size],
				name = 'softmax_weight'
				)
		self.softmax_bias = tf.get_variable(
			shape = [zh_vocab_size],
			name = 'softmax_bias'
			)

		# 连接模型
		# shape (batch_size, max word length, embedding_size)
		src_emb = tf.nn.embedding_lookup(self.src_embedding, self.src_input)
		trg_emb = tf.nn.embedding_lookup(self.trg_embedding, self.trg_input)

		# shape (batch_size, max word length, embedding_size)
		src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
		trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

		with tf.variable_scope('encoder'):
			enc_outputs, enc_state = tf.nn.dynamic_rnn(
				cell = self.encode_cell,
				inputs = src_emb,
				sequence_length = self.src_size,
				dtype = tf.float32
				)
		# dec_outputs shape = (batch_size, max_word_length, HIDDEN_SIZE)
		with tf.variable_scope('decoder'):
			dec_outputs, dec_state = tf.nn.dynamic_rnn(
				cell = self.decode_cell,
				inputs = trg_emb,
				sequence_length = self.trg_size,
				initial_state = enc_state
				)
		# dec_outputs shape = (batch_size * max_word_length, HIDDEN_SIZE)
		outputs = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
		# shape = (batch_size * max_word_length, zh_vocab_size)
		logits = tf.matmul(outputs, self.softmax_weight) + self.softmax_bias
		# shape = (batch_size * max_word_length, )
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			# (batch_size ,)
			logits = logits,
			# (batch_size * max_word_length, )
			labels = tf.reshape(self.trg_label, [-1])
			)

		# 计算损失
		# (batch_size, max word length)
		label_weights = tf.sequence_mask(
			self.trg_size, 
			maxlen = tf.shape(self.trg_label)[1],
			dtype = tf.float32
			)
		# (batch_size * max word length, )
		label_weights = tf.reshape(label_weights, [-1])
		# 计算有效长度的cost
		cost = tf.reduce_sum(loss * label_weights)
		# 计算平均每个token的cost
		self.cost_per_token = cost / tf.reduce_sum(label_weights)

		trainable_variables = tf.trainable_variables()
		batch_size = tf.shape(self.src_input)[0]
		grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
		grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)

		self.train_op = optimizer.apply_gradients(
			zip(grads, trainable_variables)
			)

	def inference(self, src_input):
		'''
		将输入src_input根据现有模型翻译
		'''
		src_size = tf.convert_to_tensor(
			value = [len(src_input)], 
			dtype = tf.int32
			)
		src_input = tf.convert_to_tensor(
			value = [src_input],
			dtype = tf.int32
			)
		src_emb = tf.nn.embedding_lookup(
			self.src_embedding,
			src_input
			)

		# 直接执行decoder，取出state
		with tf.variable_scope('encoder'):
			enc_outputs, enc_state = tf.nn.dynamic_rnn(
				self.encode_cell,
				src_emb,
				src_size,
				dtype = tf.float32
				)
		with tf.variable_scope('decoder/rnn/multi_rnn_cell'):
			# 使用一个变长的TensorArray来存储生成的句子
			init_array = tf.TensorArray(
				dtype = tf.int32,
				size = 0,
				dynamic_size = True,
				clear_after_read = False
				)

			# 填入SOS作为解码器的输入
			init_array = init_array.write(0, SOS_ID)
			# 构造loop的状态变量
			init_loop_var = (enc_state, init_array, 0)

			# 构造循环终止条件
			def continue_loop_condition(state, trg_ids, step):
				'''
				循环条件，如果解码器输出EOS或者达到最大步数，就返回False，否则返回True
				'''
				return tf.reduce_all(
					tf.logical_and(
						tf.not_equal(trg_ids.read(step), EOS_ID), 
						tf.less(step, MAX_DECODE_LENGTH - 1)
						)
					)

			# 构造循环内容
			def loop_body(state, trg_ids, step):
				'''
				循环内容，decoder模型以state和trg_ids中的输入为输入来进行传播，并且更新状态变量
				'''
				trg_input = [trg_ids.read(step)]
				# shape = (batch_size, length, embedding size)
				trg_emb = tf.nn.embedding_lookup(
					params = self.trg_embedding,
					ids = trg_input 
					)
				# (batch_size, length, HIDDEN_SIZE)
				dec_outputs, next_state = self.decode_cell(
					state = state,
					inputs = trg_emb
					)

				# (batch_size * length, HIDDEN_SIZE)
				outputs = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
				# (batch_size * length, TRG_VOCAB_SIZE)
				logits = tf.matmul(
					outputs, self.softmax_weight
					) + self.softmax_bias

				# 取出其中最大的
				next_id = tf.argmax(logits, axis = 1, output_type = tf.int32)
				trg_ids = trg_ids.write(step + 1, next_id[0])
				return next_state, trg_ids, step + 1

			# 执行tf.while_loop, 返回最终状态
			state, trg_ids, step = tf.while_loop(
				cond = continue_loop_condition,# 循环条件
				body = loop_body, # 循环body
				loop_vars = init_loop_var # 循环变量
				)
			return trg_ids.stack()




















