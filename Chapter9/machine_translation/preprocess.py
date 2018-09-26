#coding=utf-8
from parameters import *
def split_to_en_zh(source, en, zh):
    '''
    '''
    count = 0
    for line in open(source):
        # 中文
        if count % 2 == 0:
            zh.write(line)
        else:
            en.write(line)
        count += 1
# 首先将data/event_data中的数据按照中英文进行区分，放到不同的文件中，并且一一对应
en = open(en_file, 'w')
zh = open(zh_file, 'w')
split_to_en_zh(raw_train, en, zh)
split_to_en_zh(raw_dev, en, zh)
split_to_en_zh(raw_test, en, zh)
en.close()
zh.close()

# 对英文进行tokenize
import os
os.system('./mosesdecoder/scripts/tokenizer/tokenizer.perl -no-escape -l en < ./data/train.txt.en > ./data/train.txt.en.tokenized')
# 转为小写字母
os.system('./mosesdecoder/scripts/recaser/train-truecaser.perl --model ./data/trucase-model.en --corpus ./data/train.txt.en.tokenized')
os.system('./mosesdecoder/scripts/recaser/truecase.perl --model ./data/trucase-model.en < ./data/train.txt.en.tokenized > ./data/train.txt.en.tok.case')
os.system('mv ./data/train.txt.en.tok.case ' + en_file)

# 统计vocab
vocab_en_file = open(en_vocab, 'w')
vocab_zh_file = open(zh_vocab, 'w')

def generate_vocab(file_path, vocab_file):
    '''
    '''
    vocabs = {}
    for line in open(file_path):
        for word in line.strip().split(' '):
            word = word.lower()
            if word not in vocabs:
                vocabs[word] = 0
            vocabs[word] += 1

    vocabs = sorted(vocabs.items(), lambda x, y: cmp(x[1], y[1]), reverse = True)
    vocab_file.write(UNK_TOKEN + '\n')
    vocab_file.write(SOS_TOKEN + '\n')
    vocab_file.write(EOS_TOKEN + '\n')
    for vocab in vocabs:
        vocab_file.write(vocab[0] + '\n')


#统计英文
generate_vocab(en_file, vocab_en_file)

# 统计中文
generate_vocab(zh_file, vocab_zh_file)
vocab_en_file.close()
vocab_zh_file.close()

# 将文件转化为单词id，并且写入文件中
def load_vocabs(file_path):
    '''
    '''
    vocabs = []
    for line in open(file_path):
        vocabs.append(line.strip())
    return vocabs

en_vocabs = load_vocabs(en_vocab)
zh_vocabs = load_vocabs(zh_vocab)

def convert(file_path, vocabs, ids_file):
    '''
    '''
    with open(ids_file, 'w') as f:
        with open(file_path) as f1:
            for line in f1:
                words = line.strip().split(' ')
                ids = [vocabs.index(word.lower()) for word in words]
                # print ''.join(str(ids))
                f.write(''.join(str(ids))[1:][:-1] + '\n')

convert(en_file, en_vocabs, en_ids_file)
convert(zh_file, zh_vocabs, zh_ids_file)


