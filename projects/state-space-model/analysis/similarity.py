from argparse import ArgumentParser
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modelzipper.tutils import *

def analysis_cov1d_compress(fpath):
    file_names = auto_read_dir(fpath, file_prefix="passkey", file_suffix=".pkl")
    text_embedding_file = auto_read_dir(fpath, file_prefix="input_seq_embedding", file_suffix=".pkl")[0]
    text_embedding = auto_read_data(text_embedding_file)
    if text_embedding.dim() == 3:
        text_embedding = text_embedding.squeeze(0)
    for file_name in file_names:
        import pdb; pdb.set_trace()
        layer_idx = int(os.path.basename(file_name).split("-")[-1].split(".")[0])
        hidden_state = auto_read_data(file_name)

        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)

        if hidden_state.size(0) != text_embedding.size(0):
            hidden_state = hidden_state[:text_embedding.size(0)]
       
    



if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--fpath", type=str, default="analysis/inner_state")


    args = argparse.parse_args()


    analysis_cov1d_compress(args.fpath)
