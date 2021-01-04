import collections
import os
import random
import time
import re
from tqdm import tqdm
import torch
import torchtext.vocab as Vocab

import torch.utils.data as Data
import torch.nn.functional as F
import jieba
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_dataset(folder='train', data_root='C:\\Users\\46562\\Desktop\\Rnn datasets\\'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                line=f.readline()
                while  line:
                       review = line.decode('utf-8')
                       data.append([review, 1 if label == 'pos' else 0])# 评论文本字符串和01标签
                       line = f.readline()
    random.shuffle(data)
    return data

DATA_ROOT = 'C:\\Users\\46562\\Desktop\\Rnn datasets\\'
#data_root = os.path.join(DATA_ROOT, "aclImdb")
train_data, test_data = read_dataset('train', DATA_ROOT), read_dataset('test', DATA_ROOT)

# 打印训练数据中的前十个sample
# for sample in train_data[0:10]:
#     print(sample[1], '\t', sample[0][:1000])


def get_tokenized_evaluation(data):  # 将每行数据的进行空格切割,保留每个的单词
    '''
    @params:
        data: 数据的列表，列表中的每个元素为 [文本字符串，0/1标签] 二元组
    @return: 切分词后的文本的列表，列表中的每个元素为切分后的词序列
    '''
    def seg_char(text):
     pattern_char_1 = re.compile(r'([\W])')
     parts = pattern_char_1.split(text)
     parts = [p for p in parts if len(p.strip()) > 0]
     #分割中文
     pattern = re.compile(r'([\u4e00-\u9fa5])')
     chars = jieba.cut(text)                     #使用jieba中文分词器划分text文本
     chars = [w for w in chars if len(w.strip()) > 0]
     return chars



    # def tokenizer(text):
    #     return [tok.lower() for tok in text.split(' ')]

    return [seg_char(review) for review, _ in data]


def get_vocab_evaluation(data):
    '''
    @params:
        data: 同上
    @return: 数据集上的词典，Vocab 的实例（freqs, stoi, itos）
    '''
    tokenized_data = get_tokenized_evaluation(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 统计所有的数据
    return Vocab.Vocab(counter, min_freq=2)  # 构建词汇表,这里最小出现次数是2


vocab = get_vocab_evaluation(train_data)
print('# words in vocab:', len(vocab))
# print(vocab[:5])



def preprocess_evaluation(data, vocab):
    '''
    @params:
        data: 同上，原始的读入数据
        vocab: 训练集上生成的词典
    @return:
        features: 单词下标序列，形状为 (n, max_l) 的整数张量
        labels: 情感标签，形状为 (n,) 的0/1整数张量
    '''
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_evaluation(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    # 填充,这里是将每一行数据扩充500个特征的
    labels = torch.tensor([score for _, score in data])
    return features, labels


train_set = Data.TensorDataset(*preprocess_evaluation(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_evaluation(test_data, vocab))# 相当于将函数参数是函数结果
# *号语法糖,解绑参数
# 上面的代码等价于下面的注释代码
# train_features, train_labels = preprocess_imdb(train_data, vocab)
# test_features, test_labels = preprocess_imdb(test_data, vocab)
# train_set = Data.TensorDataset(train_features, train_labels)
# test_set = Data.TensorDataset(test_features, test_labels)

# len(train_set) = features.shape[0] or labels.shape[0]
# train_set[index] = (features[index], labels[index])

batch_size = 64
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter))#53个批次,每个批次64个样本
# 这是对的


