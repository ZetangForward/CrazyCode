from argparse import ArgumentParser
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modelzipper.tutils import *


def anaylsis_single_file_conv1d(embedding_path, fpath, output_path, depth=None):
    ctx_length = int(os.path.basename(fpath).split("-")[2])
    analysis_layer = int(os.path.basename(fpath).split("-")[-1].split(".")[0])
    
    if depth is None:
        depth = eval(fpath.split("/")[-2].split("-")[-1].replace("_", "."))
    
    save_file_name = f"{output_path}/conv1d_analysis_depth-{depth}_ctx-{ctx_length}_layer-{analysis_layer}.png"

    hidden_state = auto_read_data(fpath)

    if hidden_state.dim() == 3:
        hidden_state = hidden_state.squeeze(0)

        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)
        
        hidden_state = hidden_state.permute(1, 0)
    
    embedding = auto_read_data(embedding_path)

    similarity_matrix = torch.zeros(hidden_state.size(0), ctx_length).to(hidden_state.device)
    for i, h in enumerate(hidden_state):
        cos_sim = F.cosine_similarity(h.unsqueeze(0), embedding)
        similarity_matrix[i] = cos_sim



def analysis_cov1d_compress(fpath, dir=None, highlight_start=18, highlight_end=40):
    # read text embedding
    text_embedding_file = auto_read_dir(fpath, file_prefix="input_seq_embedding", file_suffix=".pkl")[0]
    text_embedding = auto_read_data(text_embedding_file) # torch.Size([1, 550, 2048])
    if text_embedding.dim() == 3:
        text_embedding = text_embedding.squeeze(0)

    # read all intermediate states
    file_names = auto_read_dir(fpath, file_prefix="passkey", file_suffix=".pkl")
    file_names = sorted(file_names, key=lambda x: int(os.path.basename(x).split("-")[2]))

    sub_files = []
    with tqdm(total=len(file_names), desc="Drawing Figure ...") as pbar:
        for file_index, file_name in enumerate(file_names):
            sub_files.append(file_name)
            
            if (file_index + 1) % 5 == 0:  # 每5个文件为一组 （因为探针了5层）
                offset = int(file_name.split("-")[2])
                
                sub_files = sorted(sub_files, key=lambda x: int(os.path.basename(x).split("-")[-1].split('.')[0]))
                fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(200, 50))
                plt.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距

                axs = axs.flatten() 
                layers, per_layer_scores = [], {}

                for idx, file_name in enumerate(sub_files):
                    layer_idx = int(os.path.basename(file_name).split("-")[-1].split(".")[0])
                    layers.append(layer_idx)

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
                    ### 真实数据的位置
                    highlight_mask = np.zeros_like(similarity_matrix_np, dtype=bool)
                    highlight_mask[:, highlight_start: highlight_end] = True
                    # 创建一个新的掩码,将highlight_mask部分设置为红色
                    green_mask = np.zeros((*similarity_matrix_np.shape, 3))
                    green_mask[highlight_mask, 1] = 1.0  # 绿色通道设为1  

                    ### 划分每个区间
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
                    
                    per_layer_scores[layer_idx] = partition_scores
                    
                    mask = np.zeros_like(similarity_matrix_np, dtype=bool)
                    for i in range(similarity_matrix_np.shape[0]):
                        mask[i, top_indices[i]] = True

                    ax = axs[idx]
                    ax.imshow(similarity_matrix_np, cmap='Reds', aspect='auto', alpha=0.3) 
                    ax.imshow(mask, cmap='hot', aspect='auto', alpha=0.9)
                    ax.imshow(green_mask, aspect='auto', alpha=0.3)  # 显示红色掩码

                    # 绘制竖线表示切分位置
                    for j in range(1, num_partitions):
                        ax.axvline(x=j * partition_length, color='yellow', linestyle='--', linewidth=20)

                    # ax.set_title(f"Layer {layer_idx}", fontsize=24)
                    # print_c(f"Layer {layer_idx} partition_scores:\n{partition_scores * 100}", "yellow")
                    
                # 调整子图位置
                plt.tight_layout()
                plt.savefig(f"analysis/figures/cosine_similarity_heatmap_offset-{offset}.png")

                # 绘制 per_layer_scores 折线图
                fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
                axs = axs.flatten()

                x_labels = ['20%', '40%', '60%', '80%', '100%']
                x_positions = np.arange(len(x_labels))
                
                for index, j in enumerate(per_layer_scores.keys()):
                    for k in range(per_layer_scores[j].shape[0]):  # 循环每一种压缩程度
                        axs[index].plot(x_positions, per_layer_scores[j][k, :], marker='o', label=f'Conv1d-State-{k}')
                    axs[index].set_xticks(x_positions)
                    axs[index].set_xticklabels(x_labels)
                    axs[index].set_xlabel('Percentage of Text')
                    axs[index].set_ylabel('Percentage of Top Indices')
                    axs[index].set_title(f'Layer {j+1}')
                    axs[index].legend()  # 将图例放在右上角

                plt.tight_layout()
                plt.savefig(f"analysis/figures/partition_scores_line_plot_offset-{offset}.png")

                sub_files.clear()  # 存储下一组数据

                pbar.update(5)   


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--anaylysis_dir", type=str, default="analysis/inner_state")
    argparse.add_argument("--fpath", type=str, default="analysis/inner_state")
    argparse.add_argument("--tokenizer_name_or_path", type=str, default="analysis/inner_state")

    args = argparse.parse_args()

    analysis_cov1d_compress(
        args.fpath, 
        highlight_start=18, 
        highlight_end=40
    )
