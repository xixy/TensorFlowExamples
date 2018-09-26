该部分完成了机器翻译模型，包括：
1. seq2seq模型
2. seq2seq + attention模型

数据部分请见data/event_data/ 下的文件，为百度翻译的结果，中文语料为ACE 2005的中文句子

使用方法请见makefile:
1. 数据预处理 make data
2. 训练模型 make train
3. 进行测试 make inference