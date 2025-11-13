
import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict, Counter, OrderedDict
import collections

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
class MyDataset(Dataset):

    def __init__(self, data_dir, split, create_data=True, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 300)
        self.min_occ = kwargs.get('min_occ', 3)

        # self.raw_data_path = os.path.join(data_dir, 'ptb.'+split+'.txt')
        self.data_file = 'data_'+split+'.csv'
        self.file_name = os.path.join(data_dir, split+'.csv')
        self.vocab_file = 'vocab.json'

        if create_data:
            print("Creating new %s llps data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']
 
    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def tokenize(self,lines, token='char'):  #@save
        """将文本行拆分为单词或字符词元"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('错误：未知词元类型：' + token)

    def _load_data(self, vocab=False):
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
            file.close()

    def _load_vocab(self): 
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        vocab_file.close()

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

 

        data = defaultdict(dict)
        df = pd.read_csv(self.file_name)
        tokens = self.tokenize(list(df['seq']))
        for  words in tokens:

            # print(words)

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length
        #self.data = pd.DataFrame(data).T
        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as d_file:
            data = json.dumps(data, ensure_ascii=False)
            d_file.write(data.encode('utf8', 'replace'))
        d_file.close()
         
        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        df = pd.read_csv(self.file_name)

        # for i, line in enumerate(list(df['seq'])):
        tokens = self.tokenize(list(df['seq']))
        tokens = [token for line in tokens for token in line]
        w2c.update(tokens)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        vocab_path = os.path.join(self.data_dir, self.vocab_file)

        with io.open(vocab_path, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))
        vocab_file.close()

        self._load_vocab()
        
class FastConvert(Dataset):
    def __init__(self, sequence_strs,create_data=True,**kwargs):
        self.sequence_strs = list(sequence_strs)
        self.max_sequence_length = kwargs.get('max_sequence_length', 300)
        self.min_occ = kwargs.get('min_occ', 3)
        if create_data:
            self._create_data()

    @classmethod
    def from_file(cls, fasta_file):
        sequence_strs = []
        buf = []
        
        def _flush_current_seq():
            nonlocal buf
            sequence_strs.append("".join(buf))
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    continue
                else:  # sequence line
                    buf.append(line.strip())

        return sequence_strs
        _flush_current_seq()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']
 
    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def tokenize(self,lines, token='char'):  #@save
        """将文本行拆分为单词或字符词元"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('错误：未知词元类型：' + token)


    def _load_vocab(self): 
        with open(os.path.join('data/', 'vocab.json'), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        vocab_file.close()

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        
    def _load_data(self, vocab=False):
        with open('temp.csv', 'r') as file:
            self.data = json.load(file)
            
    def _create_data(self):


        self._load_vocab()


        data = defaultdict(dict)
        # df = pd.read_csv(self.file_name)
        tokens = self.tokenize(self.sequence_strs)
        for  words in tokens:

            # print(words)

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length
        with io.open('temp.csv', 'wb') as d_file:
            data = json.dumps(data, ensure_ascii=False)
            d_file.write(data.encode('utf8', 'replace'))
        d_file.close()
         
        self._load_data(vocab=False)