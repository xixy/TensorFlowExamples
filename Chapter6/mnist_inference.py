#coding=utf-8
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

KEEP_PROB = 0.5

def inference(input_tensor, train, regularizer):
	'''
	input_tensor: 输入，shape = [batch, image size, image size, channel]
	train: 是否是训练模式
	regularizer: 
	'''
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable(
			name = 'weights',
			shape = [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
			initializer = tf.truncated_normal_initializer(stddev = 0.1)
			)
		conv1_biases = tf.get_variable(
			name = 'biases',
			shape = [CONV1_DEEP],
			initializer = tf.constant_initializer(0.0)
			)
		# 使用大小为5*5, 深度为32的filters，过滤器移动的步长为1，用全零填充
		# shape = (batch, 28, 28, 32)
		conv1 = tf.nn.conv2d(
			input = input_tensor,
			filter = conv1_weights,
			strides = [1, 1, 1, 1],
			padding = 'SAME'
			)
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
	
	with tf.variable_scope('layer2-pool1'):
		# shape = (batch, 14, 14, 32)
		pool1 = tf.nn.max_pool(
			value = relu1,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = 'SAME'
			)

	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable(
			name = 'weights',
			shape = [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
			initializer = tf.truncated_normal_initializer(stddev = 0.1)
			)
		conv2_biases = tf.get_variable(
			name = 'biases',
			shape = [CONV2_DEEP],
			initializer = tf.constant_initializer(0.0)
			)
		# 使用大小为5*5, 深度为64的filters，过滤器移动的步长为1，用全零填充
		# shape = (batch, 14, 14, 64)
		conv2 = tf.nn.conv2d(
			input = pool1,
			filter = conv2_weights,
			strides = [1, 1, 1, 1],
			padding = 'SAME'
			)
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

	with tf.variable_scope('layer4-pool2'):
		# shape = (batch, 7, 7, 64)
		pool2 = tf.nn.max_pool(
			value = relu2,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = 'SAME'
			)

	# 将第四层池化层的输出转化为第五层全连接层的输出,
	# (batch, 7, 7, 64)转化为向量(batch, 7 * 7 * 64)
	shape = pool2.get_shape().as_list()
	length = shape[1] * shape[2] * shape[3]
	# (batch, 7 * 7 * 64)
	print shape
	reshaped_pool2 = tf.reshape(
		pool2,
		[-1, length]
		)

	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable(
			name = 'weights',
			shape = [length, FC_SIZE],
			initializer = tf.truncated_normal_initializer(stddev = 0.1)
			)
		fc1_biases = tf.get_variable(
			name = 'biases',
			shape = [FC_SIZE],
			initializer = tf.constant_initializer(0.0)
			)
		# 只有全连接层的权重需要加入正则化
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(fc1_weights))

		# 进行计算
		# shape = (batch, 512)
		fc1 = tf.nn.relu(tf.matmul(reshaped_pool2, fc1_weights) + fc1_biases)
		# 如果是训练模式，才需要dropout
		if train:
			fc1 = tf.nn.dropout(fc1, KEEP_PROB)
		print fc1.shape

	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable(
			name = 'weights',
			shape = [FC_SIZE, NUM_LABELS],
			initializer = tf.truncated_normal_initializer(stddev = 0.1)
			)
		fc2_biases = tf.get_variable(
			name = 'biases',
			shape = [NUM_LABELS],
			initializer = tf.constant_initializer(0.0)
			)
		# 只有全连接层的权重需要加入正则化
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(fc2_weights))

		# 进行计算
		logits = tf.matmul(fc1, fc2_weights) + fc2_biases	
	return logits






