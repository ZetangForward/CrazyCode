from argparse import ArgumentParser
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modelzipper.tutils import *



def analysis_cov1d_compress(fpath):
    file_names = auto_read_dir(fpath, file_prefix="passkey", file_suffix=".pkl")
    file_names = sorted(file_names, key=lambda x: int(os.path.basename(x).split("-")[-1].split(".")[0]))
    
    text_embedding_file = auto_read_dir(fpath, file_prefix="input_seq_embedding", file_suffix=".pkl")[0]
    text_embedding = auto_read_data(text_embedding_file) # torch.Size([1, 550, 2048])
    if text_embedding.dim() == 3:
        text_embedding = text_embedding.squeeze(0)

    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(200, 50)) 
    axs = axs.flatten() 

    for idx, file_name in enumerate(file_names):
        layer_idx = int(os.path.basename(file_name).split("-")[-1].split(".")[0])
        hidden_state = auto_read_data(file_name) 
        
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)
        
        hidden_state = hidden_state.permute(1, 0)

        if hidden_state.size(-1) != text_embedding.size(-1):
            h_avg_pooled = F.avg_pool1d(hidden_state, kernel_size=2, stride=2)

        similarity_matrix = torch.zeros(4, 550).to(h_avg_pooled.device)
        for i, h in enumerate(h_avg_pooled):
            cos_sim = F.cosine_similarity(h.unsqueeze(0), text_embedding)
            similarity_matrix[i] = cos_sim
        
        similarity_matrix_np = similarity_matrix.cpu().numpy()
        top_indices = np.argpartition(similarity_matrix_np, -50, axis=1)[:, -50:]
        
        # 统计每个区间出发的关键点
         
        num_partitions = 5
        partition_length = similarity_matrix.size(-1) // num_partitions  # 每个分区的长度
        partition_scores = np.zeros((top_indices.shape[0], num_partitions))  # 存储每个分区的得分

        # 对 top_indices 进行排序
        top_indices_sorted = np.sort(top_indices, axis=1)

        # 计算每个分区的得分
        for j in range(num_partitions):
            start_idx = j * partition_length
            end_idx = start_idx + partition_length
            # 计算每个分区中top_indices的数量
            for k in range(top_indices_sorted.shape[0]):  # 遍历每个压缩的状态
                count_in_partition = np.sum((top_indices_sorted[k, :] >= start_idx) & (top_indices_sorted[k, :] < end_idx))
                partition_scores[k, j] = count_in_partition / 50.0  # 计算比例
       
        mask = np.zeros_like(similarity_matrix_np, dtype=bool)
        for i in range(similarity_matrix_np.shape[0]):
            mask[i, top_indices[i]] = True

        ax = axs[idx]
        ax.imshow(similarity_matrix_np, cmap='Reds', aspect='auto', alpha=0.3) 
        ax.imshow(mask, cmap='hot', aspect='auto', alpha=0.9)
        
        print_c(f"Layer {layer_idx} partition_scores: {partition_scores}", "yellow")
        
        ax.axis('off')
        ax.set_title(f"Layer {layer_idx}")

    # 调整子图位置
    plt.tight_layout()
    plt.savefig("analysis/figures/cosine_similarity_heatmap_combined.png")

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--fpath", type=str, default="analysis/inner_state")


    args = argparse.parse_args()


    analysis_cov1d_compress(args.fpath)
