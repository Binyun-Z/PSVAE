import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from autogluon.tabular import TabularDataset,TabularPredictor
import os
EMB_PRE_PATH = "/home/bli/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt" 
EMBED_PATH ='./ESM_embed/'
class ProteinExtractionParams:
    def __init__(
        self,
        model_location=EMB_PRE_PATH,
        fasta_file = None,
        csv_file = None,
        output_dir = None,
        toks_per_batch=10,
        repr_layers=[-1],
        include='mean',
        truncation_seq_length=512,
        nogpu=False,
    ):
        self.model_location = model_location
        self.fasta_file = fasta_file
        self.csv_file = csv_file

        self.output_dir = pathlib.Path(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.toks_per_batch = toks_per_batch
        self.repr_layers = repr_layers
        self.include = include
        self.truncation_seq_length = truncation_seq_length
        self.nogpu = nogpu


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")


    if(args.fasta_file):
        dataset = FastaBatchedDataset.from_file(args.fasta_file)
        batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
        )
        print(f"Read {args.fasta_file} with {len(dataset)} sequences")
    elif(args.csv_file):
        data_df = pd.read_csv(args.csv_file)
        
        protein_id = data_df['id']
        # class FastaBatchedDataset(object):
        #     def __init__(self, sequence_labels, sequence_strs):
        #         self.sequence_labels = list(sequence_labels)
        #         self.sequence_strs = list(sequence_strs)
        dataset = FastaBatchedDataset(data_df['id'],data_df['seq'])
        batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
        )
        print(f"Read {args.csv_file} with {len(dataset)} sequences")
    else:
        print('no file!')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include                                                                                                                                

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
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
                    result["contacts"] = conacts[i, : truncate_len, : truncate_len].clone()

                torch.save(
                    result,
                    args.output_file,
                )
def extract_embed(data_file):
    input_data  = data_file 
    output_dir = EMBED_PATH
    try:
        # 创建文件夹
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' has been created.")
    except FileExistsError:
        print(f"Folder '{output_dir}' already exists.")
    args = ProteinExtractionParams(fasta_file=input_data,output_dir=output_dir)
    run(args)
    print('Extract ESM embeddings for {}, save in {}'.format(input_data,output_dir))
def load_esm_embed(EMBED_PATH):

    EMB_LAYER = 33
    Xs = []

    for file in os.listdir(EMBED_PATH):
        fn = f'{EMBED_PATH}/{file}'
        embs = torch.load(fn)
        
        Xs.append(embs['mean_representations'][EMB_LAYER])

    Xs = torch.stack(Xs, dim=0).numpy()
    print('load esm embedding')

    return Xs
def screen(data_path):
    DATA_PATH = data_path # Path to data
    EMB_PRE_PATH = "/home/bli/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt" 
    EMBED_PATH ='./ESM_embed/'
    EMB_LAYER = 33
    extract_embed(DATA_PATH)
    embed_temp = load_esm_embed(EMBED_PATH)
    

    predicter = TabularPredictor.load('./screening_process/AutoML_ESM')
    y_pred = predicter.predict(pd.DataFrame(embed_temp))
    return(y_pred)
    
    