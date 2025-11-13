
import os
import json
import torch
import argparse
import time
from data_utlis.dataset import FastConvert
from models.model import SentenceVAE
from utils import to_var, idx2word, interpolate
from torch.utils.data import DataLoader
from screening_process.screen import screen
def main(args):
    
    seqs = ['KKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIHFGSD',
       'SKGPGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGY'
       ]
    with open(args.data_dir+'/vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    
    # batch = next(iter(DataLoader(FastConvert(seqs),batch_size=2,pin_memory=torch.cuda.is_available())))
    # for k, v in batch.items():
    #     if torch.is_tensor(v):
    #         batch[k] = to_var(v)


    
    print('-------------Screening procedures based on ESM and Autogluon---------')
    num_screen = 0
    
    screened_seq = []
    while(num_screen<1000):
        
        
        samples_sa, z = model.inference(n=args.num_samples)
        print('----------SAMPLES----------')
        # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        
        # logp, mean, logv, z = model(batch['input'], batch['length'])
        # z = z.cpu().detach().numpy()
        # z_i = to_var(torch.from_numpy(interpolate(start=z[0], end=z[1], steps=32)).float())
    
        # samples, _ = model.inference(z=z_i)
        # print('-------INTERPOLATION GENERATION-------')
        # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        temp_data_path = os.path.join(args.data_dir,'results/','temp_generate_'+str(args.num_samples)+'.fasta')
        temp_seqlist = []
        with open(temp_data_path,'w') as rf:
            for i,sp in enumerate(idx2word(samples_sa, i2w=i2w, pad_idx=w2i['<pad>'])):
                sp = sp.replace(" ", "").replace("<eos>", "")
                temp_seqlist.append(sp)
                rf.write('>'+str(i)+'\n'+sp+'\n')
                
        print('-------SCREENing-------')
        predicter = list(screen(temp_data_path))
        selected_seq= [string for flag, string in zip(predicter, temp_seqlist) if flag == 1]
        print(selected_seq)
        screened_seq.extend(selected_seq)
        num_screen = num_screen+predicter.count(1)

    
    print(screened_seq)
    with open(os.path.join(args.data_dir,'results/','screen_num_samples_50_generate_1000'+'.fasta'),'w') as rf:
        for i,sp in enumerate(screened_seq):
            temp_seqlist.append(sp)
            rf.write('>'+str(i)+'\n'+sp+'\n')

    



class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self
inference_cfg = Bunch(
    
    
    load_checkpoint = './checkpoints/2023-Dec-05-03:19:06/E499.pytorch',    #The path to the directory where PTB data is stored, and auxiliary data files will be stored.
    num_samples = 50,         #生成的新数据量
    
    data_dir = './data',
    max_sequence_length = 150, #Specifies the cut off of long sentences.
    embedding_size = 256,
    rnn_type = 'gru', #rnn_type Either 'rnn' or 'gru'.
    hidden_size = 256, #hidden_size
    word_dropout = 0, #word_dropout Word dropout applied to the input of the Decoder which means words will be replaced by <unk> with a probability of word_dropout.
    embedding_dropout = 0.3, #embedding_dropout Word embedding dropout applied to the input of the Decoder.

    latent_size = 64, #latent_size
    num_layers = 1, #num_layers
    bidirectional = False, #bidirectional
)
if __name__ == '__main__':
    main(inference_cfg)
