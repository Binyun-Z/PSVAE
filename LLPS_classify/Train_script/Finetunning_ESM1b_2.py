
DATA_PATH = "../dataset/dataset2.0/" # Path to data
EMB_PRE_PATH = "/home/bli/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt" 
EMBED_PATH =DATA_PATH+'/ESM_embed/'
EMB_LAYER = 33
LOG_FILE = "./ESM_finetuning/Finetuning_ESM_log/Finetunning_ESM1b_2.log"
CHECKPOINT_PATH = './ESM_finetuning/checkpoint_esm2_2.0'
import os
import pandas as pd
import pathlib
import pandas as pd
import torch
from torch import nn 
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from model import LayerNormNet


class ProteinExtractionParams:
    def __init__(
        self,
        model_location=EMB_PRE_PATH,
        fasta_file = None,
        csv_file = None,
        toks_per_batch = 512,
        repr_layers=[33],
        include='mean',
        truncation_seq_length = 512,
        nogpu=False,
    ):
        self.model_location = model_location
        self.fasta_file = fasta_file
        self.csv_file = csv_file
        self.toks_per_batch = toks_per_batch
        self.repr_layers = repr_layers
        self.include = include
        self.truncation_seq_length = truncation_seq_length
        self.nogpu = nogpu
args = ProteinExtractionParams()

esm1b,alphabet = pretrained.load_model_and_alphabet(args.model_location)
from sklearn.model_selection import train_test_split
def CreateDataSet(csv_list):
    dflist = []
    for csv_file  in csv_list:
        dflist.append(pd.read_csv(csv_file))
    data_df = pd.concat(dflist)
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df,val_df = train_test_split(data_df,test_size=0.2)
    # class FastaBatchedDataset(object):
    #     def __init__(self, sequence_labels, sequence_strs):
    #         self.sequence_labels = list(sequence_labels)
    #         self.sequence_strs = list(sequence_strs)
    train_dataset = FastaBatchedDataset(train_df['label'],train_df['seq'])
    train_batches = train_dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=train_batches
    )
    
    val_dataset = FastaBatchedDataset(val_df['label'],val_df['seq'])
    val_batches = val_dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=val_batches
    )
    # print(f"Read {args.csv_file} with {len(dataset)} sequences")
    return train_dataloader,val_dataloader

csvlist = [DATA_PATH+'positive_train_422.csv',DATA_PATH+'negative_train_3307.csv']
train_dataloader,val_dataloader = CreateDataSet(csvlist)


class fine_tunning_esm(nn.Module):
  def __init__(self,args = None):
    super(fine_tunning_esm, self).__init__()
    self.esm1b,self.alphabet = pretrained.load_model_and_alphabet(args.model_location)
    self.model = LayerNormNet(hidden_dim=512,out_dim=2)

  def forward(self, strs,toks):
    return_contacts = "contacts" in args.include
    out = self.esm1b(toks, repr_layers=args.repr_layers, return_contacts=return_contacts)

    representations = {
        layer: t for layer, t in out["representations"].items()
    }
    if return_contacts:
        contacts = out["contacts"].to(device="cpu") 
    x = []   
    for i, tok in enumerate(toks):
        result = {}
        truncate_len = min(args.truncation_seq_length, len(strs[i]))
        if "per_tok" in args.include:
            result["representations"] = {
                layer: t[i, 1 : truncate_len + 1].clone()
                for layer, t in representations.items()
            }
        if "mean" in args.include:
            result["mean_representations"] = {
                layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                for layer, t in representations.items()
            }
        if "bos" in args.include:
            result["bos_representations"] = {
                layer: t[i, 0].clone() for layer, t in representations.items()
            }
        if return_contacts:
            result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()
        x.append(result['mean_representations'][33])
    x = torch.vstack((*x,))
    out = self.model(x)
    return out


import utils
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = utils.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_iter):
            metric.add(accuracy(net(strs,toks), labels), labels.numel())
    return metric[0] / metric[1]
def evaluate_accuracy_gpu(net, data_iter, devices):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    net.eval()  # 设置为评估模式
    # 正确预测的数量，总预测的数量
    metric = utils.Accumulator(2)
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_iter):
            labels = torch.tensor(labels)
            labels = labels.to(devices[0])
            toks = toks.to(devices[0])
            metric.add(accuracy(net(strs,toks), labels), labels.numel())
    return metric[0] / metric[1]

def train_batch(net, strs,X, y, loss, trainer, devices):

    X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
                
    pred = net(strs,X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def main(net, train_dataloader, val_dataloader, loss, trainer, num_epochs, start_epoch, devices = utils.try_all_gpus()):
    #####################建立一个日志用来储存训练进度
    import logging
    #记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #处理器
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    #默认使用logger的级别
    fileHandler = logging.FileHandler(filename = LOG_FILE)
    #格式
    formatter = logging.Formatter("%(asctime)s|%(message)s",)
    #给处理器设置格式
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    #记录器设置处理器
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    #########################################
    """用多GPU进行模型训练"""
    timer, num_batches = utils.Timer(), len(train_dataloader)
    # animator = utils.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
    #                         legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    best_acc = 0
    for epoch in range(start_epoch, num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = utils.Accumulator(4)
        print(f'epoch {epoch + 1} is processing')
        
        
        
        for batch_idx, (labels, strs, toks) in enumerate(train_dataloader):
            
            timer.start()
            labels = torch.tensor(labels)
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            l, acc = train_batch(
                net,strs, toks, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            # if (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
            #     animator.add(epoch + (batch_idx + 1) / num_batches,
            #                  (metric[0] / metric[2], metric[1] / metric[3],
            #                   None))


        test_acc = evaluate_accuracy_gpu(net, val_dataloader,devices)
        # animator.add(epoch + 1, (None, None, test_acc))
        logger.info(f'epoch {epoch + 1}/{num_epochs}:'\
            f' loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}'\
            f' | {metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(devices)}') 

        if not os.path.isdir(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)

        if test_acc >= best_acc:
            best_acc = test_acc
            checkpoint = {
              "net": net.module.state_dict(),
              'optimizer':trainer.state_dict(),
              "epoch": epoch
              }
            torch.save(checkpoint, CHECKPOINT_PATH+'/bestmodel.pkl')



net = fine_tunning_esm(args=args)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

trainer = torch.optim.Adam([{"params": net.esm1b.parameters(), "lr": 1e-7}],
                            lr = 1e-7, weight_decay=1e-3)
net.train()
devices = utils.try_all_gpus()
loss = nn.CrossEntropyLoss(reduction="none").cuda()

#################################################加载checkpoint模块部分###################################################
RESUME = False
num_epochs = 100
start_epoch = 0

if RESUME:
    path_checkpoint = "./ESM_finetuning/checkpoint_esm2.0/bestmodel.pkl"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    net.cuda()
    trainer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch 
    # lr_schedule.load_state_dict(checkpoint['lr_schedule'])

main(net, train_dataloader, val_dataloader, loss, trainer, num_epochs, start_epoch, devices)

