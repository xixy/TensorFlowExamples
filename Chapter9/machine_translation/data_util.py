#coding=utf-8
import random
pad_token = 0
from parameters import *

def make_batch(src_ids_file, trg_ids_file, batch_size):
	'''
	Args:

	Returns:
		src_input: (batch_size, max word length)
		src_size: (batch_size, )
		trg_input: (batch_size, max word length)
		trg_label: (batch_size, max word length, )
		trg_size: (batch_size, )
	'''
	src_lines = []
	trg_lines = []
	# 读取数据
	for line in open(src_ids_file):
		# print line
		src_lines.append([int(x) for x in line.strip().split(', ')])

	for line in open(trg_ids_file):
		trg_lines.append([int(x) for x in line.strip().split(', ')])

	random.shuffle(src_lines)
	random.shuffle(trg_lines)

	# 取出batch
	i = 0
	while i < len(src_lines):
		src_batch = src_lines[i:i+batch_size]
		trg_batch = trg_lines[i:i+batch_size]
		i += batch_size

		# 处理src_input和src_size
		src_input, src_size = padding(src_batch)
		# print src_input
		# 处理trg_input和trg_label, trg_size
		trg_input = [[SOS_ID] + data  for data in trg_batch]
		trg_label = [data + [EOS_ID] for data in trg_batch]
		trg_input, trg_size = padding(trg_input)
		trg_label, _ = padding(trg_label)

		yield (src_input, src_size, trg_input, trg_label, trg_size)





	# 按照batch size进行

def padding(datas):
	'''
	没有做clip，只做了padding
	'''
	# 先找到max word length
	max_word_length = 0
	size = []
	for data in datas:
		size.append(len(data))
		max_word_length = len(data) if len(data) > max_word_length else max_word_length
	
	return [data + [pad_token] * (max_word_length - len(data))for data in datas], size

if __name__ == '__main__':
	en_ids_file = './data/train.txt.en.id'
	zh_ids_file = './data/train.txt.zh.id'
	BATCH_SIZE = 100
	for src_input, src_size, trg_input, trg_label, trg_size in make_batch(en_ids_file, zh_ids_file, BATCH_SIZE):
		print src_input, src_size, 
		print trg_input, trg_label, trg_size
		# break