# llama
similarity_portion1 = [0.0047,0.0205,0.0603,0.1271,0.19,0.2262,0.1962,0.1209,0.0436,0.0097]  
toxicity_gen1 = [0.5038,0.4737,0.4762,0.4734,0.4792,0.4776,0.4757,0.4881,0.5163,0.5539]  
toxicity_wo_gen1 = [0.1214,0.1228,0.1614,0.1883,0.2298,0.2729,0.3211,0.3872,0.4779,0.5467]  


# gpt2-xl
similarity_portion2 = [0.0029,0.0194,0.0541,0.1179,0.1933,0.2371,0.2113,0.1229,0.0345,0.0054]
toxicity_gen2 = [0.4971,0.4775,0.4743,0.4811,0.4753,0.4767,0.4844,0.4812,0.5235,0.6398]
toxicity_wo_gen2 = [0.1293,0.1169,0.1528,0.1955,0.2302,0.2687,0.319,0.3662,0.4716,0.6385]


# gedi
similarity_portion3 = [0.0116,0.0589,0.1636,0.257,0.2418,0.1599,0.0806,0.0221,0.0027,0]
toxicity_gen3 = [0.4477,0.4497,0.4566,0.4475,0.4167,0.3684,0.331,0.3234,0.3385,0]
toxicity_wo_gen3 = [0.0332,0.0336,0.0406,0.0491,0.0643,0.0965,0.1357,0.1801,0.2578,0]


# dexperts
similarity_portion4 = [0.0091,0.0371,0.0983,0.1737,0.2323,0.2289,0.1504,0.0592,0.0087,0.0002]
toxicity_gen4 = [0.4456,0.4505,0.4319,0.4181,0.4048,0.3912,0.3795,0.3756,0.3925,0.608]
toxicity_wo_gen4 = [0.0389,0.0491,0.0557,0.0655,0.0839,0.1019,0.1304,0.1719,0.2677,0.4762]


# SGEAT
similarity_portion5 = [0.0322, 0.2034, 0.4148, 0.3063, 0.0418]
toxicity_wo_gen5 = [0.082, 0.1382, 0.2121, 0.304, 0.4749]


# radio figure
llama_similarity_value = sum(similarity_portion1[6:])
llama_tocicity = sum(toxicity_wo_gen1) / len(toxicity_wo_gen1)
llama_ppl = 6.4294

gpt_similarity_value = sum(similarity_portion2[6:])
gpt_tocicity = sum(toxicity_wo_gen2) / len(toxicity_wo_gen2)
gpt_ppl = 7.8481

gedi_similarity_value = sum(similarity_portion3[6:])
gedi_tocicity = sum(toxicity_wo_gen3) / len(toxicity_wo_gen3)
gedi_ppl = 68.8998

dexpert_similarity_value = sum(similarity_portion4[6:])
dexpert_tocicity = sum(toxicity_wo_gen4) / len(toxicity_wo_gen4)
dexpert_ppl = 16.8638

sgeat_similarity_value = sum(similarity_portion4[4:])
sgeat_tocicity = sum(toxicity_wo_gen5) / len(toxicity_wo_gen5)
sgeat_ppl = 5.4672


import numpy as np
import matplotlib.pyplot as plt

SIMs = np.array([llama_similarity_value, gpt_similarity_value, gedi_similarity_value, dexpert_similarity_value, sgeat_similarity_value])
norm_SIMs = SIMs / np.sum(SIMs)

PPLs = np.array([llama_ppl, gpt_ppl, gedi_ppl, dexpert_ppl, sgeat_ppl])
inverse_PPLs = 1 / PPLs
norm_PPLs = inverse_PPLs / np.sum(inverse_PPLs)

TOXICs = np.array([llama_tocicity, gpt_tocicity, gedi_tocicity, dexpert_tocicity, sgeat_tocicity])
inverse_TOXICs = 1 / TOXICs
norm_TOXICs = inverse_TOXICs / np.sum(inverse_TOXICs)

print(norm_SIMs)
print(norm_PPLs)
print(norm_TOXICs)


# 模型数据
data = {
    'LLaMA2-7B': {
        'similarity': norm_SIMs[0],
        'toxicity': norm_TOXICs[0],
        'ppl': norm_PPLs[0],
    },
    'GPT2-XL': {
        'similarity': norm_SIMs[1],
        'toxicity': norm_TOXICs[1],
        'ppl': norm_PPLs[1],
    },
    'Gedi': {
        'similarity': norm_SIMs[2],
        'toxicity': norm_TOXICs[2],
        'ppl': norm_PPLs[2],
    },
    'DExpert': {
        'similarity': norm_SIMs[3],
        'toxicity': norm_TOXICs[3],
        'ppl': norm_PPLs[3],
    },
    'SGEAT': {
        'similarity': norm_SIMs[4],
        'toxicity': norm_TOXICs[4],
        'ppl': norm_PPLs[4],
    }
}

print(data)


# 设置雷达图的标签  
labels=np.array(['Non-Toxicity', 'Coherence', 'Relevance'])  
  
# 计算雷达图的角度，确保图形是三角形  
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()  
angles += angles[:1]  # 闭合雷达图  

print(angles)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  

# 绘制雷达图  
for model_name, attrs in data.items():  
    stats = np.array([attrs['toxicity'], attrs['ppl'], attrs['similarity']])  # Coherence是ppl的倒数  
    stats = np.concatenate((stats,[stats[0]]))  # 闭合雷达图  
    if model_name == 'LLaMA2-7B' or model_name == 'GPT2-XL':
        ax.plot(angles, stats, label=model_name, linestyle='--')  
    else:
        ax.plot(angles, stats, label=model_name)
    ax.fill(angles, stats, alpha=0.25)  
  
# 添加标签，不包括闭合的额外角度  
ax.set_thetagrids([])

# # 设置雷达图的标签位置  
for label, angle in zip(labels, angles[:-1]): 
    if label == "Non-Toxicity":
        angle += 0.20
        cc = 0.30
    elif label == "Relevance":
        angle += 0.32
        cc = 0.35
    else:
        angle += 0.36
        cc = 0.36
    ax.text(angle, cc, label, ha='center', va='center', fontsize=18, fontname= 'DejaVu Sans Mono', fontweight=600, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=1))

ax.grid(True)
ax.set_rgrids([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], labels=[])

# 设置雷达图为三角形  
ax.set_ylim(0)  # 可以通过设置y轴的下限来确保图形是三角形  

# 添加图例  
plt.legend(loc='upper right', bbox_to_anchor=(0.12, 0.12), prop={'family': 'DejaVu Sans Mono', 'weight': 600, 'size': 16})  

plt.savefig('save_figs/radar_chart_no_legend.pdf', bbox_inches='tight', pad_inches=0.1)
