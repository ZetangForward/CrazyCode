import matplotlib.pyplot as plt  
import numpy as np  

# golbal config
plt.rcParams['font.weight'] = 'bold'
bar_width = 0.17
opacity = 0.7
# fig, ax = plt.subplots(figsize=(8,6))  
similarity = [0.2, 0.4, 0.6, 0.8, 1.0]  

# llama
similarity_portion1 = [0.0253,0.1875,0.4162,0.3172,0.0534]
toxicity_gen1 = [0.4793,0.4743,0.4784,0.4804,0.5232]
toxicity_wo_gen1 = [0.1225,0.1796,0.2532,0.3463,0.4905]

# gpt2-xl
similarity_portion2 = [0.0224,0.1721,0.4305,0.3342,0.04]
toxicity_gen2 = [0.4801,0.479,0.476,0.4832,0.5394]
toxicity_wo_gen2 = [0.1186,0.1821,0.2514,0.3364,0.4944]

# dexpert
similarity_portion3 = [0.0463,0.272,0.4613,0.2097,0.0089]
toxicity_gen3 = [0.4495,0.4231,0.398,0.3784,0.398]
toxicity_wo_gen3 = [0.0471,0.062,0.0929,0.1421,0.273]

# gedi
similarity_portion4 = [0.0705,0.4207,0.4018,0.1027,0.0027]
toxicity_gen4 = [0.4494,0.451,0.3975,0.3294,0.3385]
toxicity_wo_gen4 = [0.0335,0.0458,0.0771,0.1452,0.2578]

# llama adapter
similarity_portion5 = [0.0322,0.2034,0.4148,0.3063,0.0418]
toxicity_gen5 = [0.4581,0.4717,0.4657,0.4903,0.5581]
toxicity_wo_gen5 = [0.0602,0.1314,0.2229,0.3822,0.532]

# llama_mask
similarity_portion1_mask = [0.0374,0.258,0.4224,0.2327,0.0484]
toxicity_gen1_mask = [0.1179,0.1465,0.1682,0.1929,0.1944]
toxicity_wo_gen1_mask = [0.0876,0.1166,0.1481,0.1809,0.2023]

# gpt2-xl_mask
similarity_portion2_mask = [0.0244,0.2384,0.4553,0.2422,0.0392]
toxicity_gen2_mask = [0.1525,0.1631,0.1782,0.2081,0.2043]
toxicity_wo_gen2_mask = [0.1246,0.1379,0.1579,0.1991,0.2085]

# dexpert_mask
similarity_portion3_mask = [0.0444,0.3574,0.4658,0.1256,0.0054]
toxicity_gen3_mask = [0.108,0.114,0.1147,0.1252,0.167]
toxicity_wo_gen3_mask = [0.0439,0.0505,0.0637,0.0865,0.1495]

# gedi_mask
similarity_portion4_mask = [0.0686,0.3963,0.439,0.0924,0.0015]
toxicity_gen4_mask = [0.1438,0.1325,0.1307,0.1457,0.1369]
toxicity_wo_gen4_mask = [0.0594,0.0718,0.0892,0.1167,0.128]

# llama adapter_mask
similarity_portion5_mask = [0.0526,0.1634,0.2111,0.1723,0.3978]
toxicity_gen5_mask = [0.0891,0.1141,0.1301,0.1466,0.1776]
toxicity_wo_gen5_mask = [0.0358,0.0578,0.0876,0.1247,0.1755]


index1 = np.arange(len(similarity_portion1))
  
fig, ax1 = plt.subplots(figsize=(8,6))  
  
bar1 = plt.bar(index1, similarity_portion1, bar_width, alpha=opacity, color='b', label='LLaMA2-7B')  
bar2 = plt.bar(index1 + bar_width, similarity_portion2, bar_width, alpha=opacity, color='r', label='GPT2-XL')  
bar3 = plt.bar(index1 + 2*bar_width, similarity_portion3, bar_width, alpha=opacity, color='g', label='Gedi')  
bar4 = plt.bar(index1 + 3*bar_width, similarity_portion4, bar_width, alpha=opacity, color='c', label='DExperts')  
bar5 = plt.bar(index1 + 4*bar_width, similarity_portion5, bar_width, alpha=opacity, color='orange', label='SGEAT') 


ax1.set_xlabel('Semantic Similarity', fontsize=17, fontweight='bold')  
ax1.set_ylabel('Proportion', fontsize=17, fontweight='bold')  
plt.xticks([i +2*bar_width for i in index1], ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=17)
plt.yticks(fontsize=17)  
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray')
ax1.legend(fontsize=14, loc='upper left')  
  
plt.tight_layout() 
plt.savefig("save_figs/half_semantic_proportion_all.png") 
# plt.show()  
