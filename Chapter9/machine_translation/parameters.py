#coding=utf-8
# raw data部分
raw_train = './data/event_data/train.dat'
raw_dev = './data/event_data/dev.dat'
raw_test = './data/event_data/test.dat'
en_file = './data/train.txt.en'
zh_file = './data/train.txt.zh'

en_vocab = './data/vocab.en'
zh_vocab = './data/vocab.zh'

en_ids_file = './data/train.txt.en.id'
zh_ids_file = './data/train.txt.zh.id'
HIDDEN_SIZE = 1024 # LSTM 隐藏层网络
NUM_LAYERS = 2
en_vocab_size = 12633
zh_vocab_size = 18954
BATCH_SIZE = 10
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARED_EMD_AND_SOFTMAX = True
CHECKPOINT_PATH = './model/seq2seq_ckpt'
LEARNING_RATE = 0.9

CHECKPOINT_RESTORE_PATH = './model/seq2seq_ckpt-80'

UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

MAX_DECODE_LENGTH = 100