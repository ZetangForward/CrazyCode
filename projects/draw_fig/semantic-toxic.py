import matplotlib.pyplot as plt  
import numpy as np  

# golbal config
# plt.rcParams['font.weight'] = 'bold'
bar_width = 0.25
opacity = 0.7
fig, ax = plt.subplots(figsize=(8,6))  

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
similarity_portion5 = [0.033,0.1401,0.226,0.2524,0.3467]
toxicity_gen5 = [0.4581,0.4717,0.4657,0.4903,0.5581]
toxicity_wo_gen5 = [0.082,0.1382,0.2121,0.304,0.4749]

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
toxicity_wo_gen5_mask = [0.0756,0.0966,0.126,0.1673,0.2228]


index = np.arange(len(similarity))  

toxicity_wo_gen2 = [min(0.25, i) for i in toxicity_wo_gen2]
plt.plot(index, toxicity_wo_gen2, linewidth=2, marker='o', linestyle='-', alpha=opacity, color='orange', label='GPT2-XL')  
toxicity_wo_gen3 = [min(0.25, i) for i in toxicity_wo_gen3]
plt.plot(index+bar_width, toxicity_wo_gen3, linewidth=2, marker='o', linestyle='-', alpha=opacity, color='g', label='Gedi')  
toxicity_wo_gen4 = [min(0.25, i) for i in toxicity_wo_gen4]
plt.plot(index+2*bar_width, toxicity_wo_gen4, linewidth=2, marker='o', linestyle='-', alpha=opacity, color='c', label='DExperts')  
# line9 = plt.plot(index+2*bar_width, toxicity_wo_gen5, linewidth=2, marker='o', linestyle='--', alpha=opacity, color='orange', label='SGEAT')

plt.xlabel('Semantic Similarity', fontsize=16, fontname= 'DejaVu Sans Mono', fontweight=600,bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=1))  
# plt.ylabel('Generation Toxicity', fontsize=11, fontname= 'DejaVu Sans Mono', fontweight=600,bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=1))  
plt.ylim(0, 0.34)
plt.xticks(index+bar_width, ['<0.2', '0.4', '0.6', '0.8', '>0.8'], fontsize=16, fontname= 'DejaVu Sans Mono', fontweight=600)  
ax.set_yticks([0.0, 0.1, 0.2, 0.25])
ax.set_yticklabels(["0.0", "0.1", "0.2", ">0.25"])
plt.yticks(fontsize=16, fontname= 'DejaVu Sans Mono', fontweight=600) 
ax.yaxis.grid(True, linewidth=0.5, color='gray')
plt.legend(loc='upper left',prop={'family': 'DejaVu Sans Mono', 'weight': 600, 'size': 14})  

ax2 = ax.twinx()  
# ax2.set_ylabel('Generation Toxicity[MASK]', fontsize=11, fontname= 'DejaVu Sans Mono', fontweight=600,bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=1))  
plt.ylim(0, 0.34)
line2 = plt.bar(index, toxicity_wo_gen2_mask, bar_width, alpha=opacity, color='orange', label='GPT2-XL[MASK]')
line3 = plt.bar(index+bar_width, toxicity_wo_gen3_mask, bar_width, alpha=opacity, color='g', label='Gedi[MASK]')  
line4 = plt.bar(index+2*bar_width, toxicity_wo_gen4_mask, bar_width, alpha=opacity, color='c', label='DExperts[MASK]')  
# line5 = plt.bar(index+2*bar_width, toxicity_wo_gen5_mask, bar_width, alpha=opacity, color='orange', label='SGEAT (mask)')

ax2.set_yticks([0.0, 0.1, 0.2, 0.25])
ax2.set_yticklabels(["0.0", "0.1", "0.2", ">0.25"])
plt.yticks(fontsize=16, fontname= 'DejaVu Sans Mono', fontweight=600) 
plt.legend(loc='upper right', prop={'family': 'DejaVu Sans Mono', 'weight': 600, 'size': 14}) 

plt.tight_layout()  
plt.savefig(r'save_figs/pre_semantic_no_legend.pdf',)