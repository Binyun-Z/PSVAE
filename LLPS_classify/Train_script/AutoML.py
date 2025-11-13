import numpy as np
import pandas as pd
import os
import torch

Embed = 'ESM'

DATA_PATH = '../dataset/dataset2.0/'


def load_esm_embed(csv_file,embed_type):
    EMBED_PATH = DATA_PATH+embed_type+'_embed/'
    EMB_LAYER = 33
    Xs = []
    ys = []
    Embed_PATH = EMBED_PATH+csv_file.split('.')[0]
    data_df =  pd.read_csv(DATA_PATH+csv_file)
    for index, row in data_df.iterrows():
        id = row['id']
        label = row['label']

        fn = f'{Embed_PATH}/{id}.pt'
        embs = torch.load(fn)
        
        Xs.append(embs['mean_representations'][EMB_LAYER])
        ys.append(label)
    Xs = torch.stack(Xs, dim=0).numpy()
    print('load {} esm embedding'.format(csv_file))
    print(len(ys))
    print(Xs.shape)
    return Xs,ys

def load_embed(csv_file,embed_type):
    EMBED_PATH = DATA_PATH+embed_type+'_embed/'
    Embed_PATH = EMBED_PATH+csv_file.split('.')[0]+'_embeds.npy'
    data_df =  pd.read_csv(DATA_PATH+csv_file)
    ys = data_df['label']
    Xs = np.load(Embed_PATH)
    print('load {} {}_embed embedding from {}'.format(csv_file,embed_type,Embed_PATH))
    print(len(ys))
    print(Xs.shape)
    return Xs,ys








def embedding(csv_file,embed= 'ESM'):
    if embed=='ESM':Xs,ys = load_esm_embed(csv_file,embed)
    else:Xs,ys = load_embed(csv_file,embed)
    return Xs,ys


def Create_dataset(embed='ESM'):
    positive_data = 'positive_train_422.csv'
    negative_data = 'negative_train_3307.csv'
    Xs_p,ys_p = embedding(positive_data,embed)
    Xs_n,ys_n = embedding(negative_data,embed)
    p_df = pd.DataFrame(Xs_p)
    p_df['label'] = list(ys_p)
    n_df = pd.DataFrame(Xs_n)
    n_df['label'] = list(ys_n)
    return p_df,n_df
    



from autogluon.tabular import TabularDataset,TabularPredictor
def AutoML(samlpe = None):
    embed_types = ['ESM','ProtBert_bfd','ProtBert','T5','UniRep']
    if samlpe == 'Oversampling':
        for embed in embed_types:
            p_df,n_df = Create_dataset(embed)
            p_df = pd.concat([p_df]*8)
            df = pd.concat([p_df,n_df])
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            df = TabularDataset(df)
            TabularPredictor(label="label",path='AutoML_result_0609/AutoML_{}_Oversampling'.format(embed)).fit(df,num_bag_folds=5)

AutoML(samlpe='Oversampling')