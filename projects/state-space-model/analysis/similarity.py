from argparse import ArgumentParser
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modelzipper.tutils import *



def analysis_cov1d_compress(fpath):
    file_names = auto_read_dir(fpath, file_prefix="passkey", file_suffix=".pkl")
    text_embedding_file = auto_read_dir(fpath, file_prefix="input_seq_embedding", file_suffix=".pkl")[0]
    text_embedding = auto_read_data(text_embedding_file) # torch.Size([1, 550, 2048])
    if text_embedding.dim() == 3:
        text_embedding = text_embedding.squeeze(0)
    for file_name in file_names:
        import pdb; pdb.set_trace()
        
        layer_idx = int(os.path.basename(file_name).split("-")[-1].split(".")[0])
        hidden_state = auto_read_data(file_name)  # torch.Size([1, 4096, 4])
        
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)

        hidden_state = hidden_state.permute(1, 0)

        if hidden_state.size(-1) != text_embedding.size(-1):
            h_avg_pooled = F.avg_pool1d(hidden_state, kernel_size=2, stride=2)  # 4, 2048

        similarity_matrix = torch.zeros(4, 550).to(h_avg_pooled.device)  # for recording the similarity between each hidden state and the text embedding
        for i, h in enumerate(h_avg_pooled):
            cos_sim = F.cosine_similarity(h.unsqueeze(0), text_embedding)
            similarity_matrix[i] = cos_sim

        similarity_matrix_np = similarity_matrix.cpu().numpy()

        # 绘制热力图
        plt.imshow(similarity_matrix_np, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Text Embedding Index')
        plt.ylabel('Hidden State Index')
        plt.savefig(f"analysis/figures/cosine_similarity_heatmap_layer_{layer_idx}.png")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--fpath", type=str, default="analysis/inner_state")


    args = argparse.parse_args()


    analysis_cov1d_compress(args.fpath)
