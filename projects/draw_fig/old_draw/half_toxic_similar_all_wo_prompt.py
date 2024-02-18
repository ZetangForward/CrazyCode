import matplotlib.pyplot as plt  
import numpy as np  

# golbal config
plt.rcParams['font.weight'] = 'bold'
fig, ax = plt.subplots()  
similarity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

# llama_mask
similarity_portion1_mask = [0.0056,0.0318,0.0893,0.1686,0.2131,0.2093,0.1536,0.0791,0.0361,0.0122]
toxicity_gen1_mask = [0.0965,0.1217,0.1412,0.1493,0.1611,0.1754,0.19,0.1984,0.1946,0.194]
toxicity_wo_gen1_mask = [0.0776,0.0893,0.1107,0.1197,0.1394,0.1569,0.1759,0.1907,0.2002,0.2086]

# gpt2-xl_mask
similarity_portion2_mask = [0.0032,0.0212,0.0809,0.1575,0.2303,0.2249,0.1616,0.0805,0.0314,0.0077]
toxicity_gen2_mask = [0.1479,0.1532,0.1634,0.163,0.1719,0.1847,0.1974,0.2295,0.2092,0.1845]
toxicity_wo_gen2_mask = [0.1282,0.124,0.1348,0.1395,0.1492,0.1668,0.1867,0.2241,0.2102,0.2016]

# gedi_mask
similarity_portion3_mask = [0.014,0.0546,0.1551,0.2411,0.2585,0.1805,0.075,0.0174,0.0014,0.0001]
toxicity_gen3_mask = [0.1524,0.1416,0.1344,0.1312,0.1287,0.1334,0.1425,0.1597,0.1414,0.0832]
toxicity_wo_gen3_mask = [0.0536,0.0609,0.0642,0.0767,0.0836,0.0973,0.1142,0.1272,0.1304,0.0998]

# dexpert_mask
similarity_portion4_mask = [0.0067,0.0376,0.126,0.2314,0.2644,0.2013,0.0985,0.027,0.0048,0.0005]
toxicity_gen4_mask = [0.1172,0.1064,0.1134,0.1143,0.1141,0.1155,0.1213,0.1395,0.1721,0.1191]
toxicity_wo_gen4_mask = [0.0537,0.0421,0.0472,0.0523,0.0602,0.0683,0.0819,0.1031,0.1521,0.125]

bar_width = 0.2
opacity = 0.7
  
index = np.arange(len(similarity))  

# print(index)
# print(index + bar_width*3)

bar1 = plt.bar(index, toxicity_wo_gen1, bar_width,  
alpha=opacity,  
color='blue',  
# edgecolor='g', 
# hatch='\\\\', 
label='LLaMA-7B')  

bar2 = plt.bar(index + bar_width, toxicity_wo_gen2, bar_width,  
alpha=opacity,  
color='red',  
# edgecolor='r',
# hatch='xx', 
label='GPT2-XL') 

bar3 = plt.bar(index + bar_width*2, toxicity_wo_gen3, bar_width,  
alpha=opacity,  
color='g',  
# edgecolor='g', 
# hatch='\\\\', 
label='Gedi')  

bar4 = plt.bar(index + bar_width*3, toxicity_wo_gen4, bar_width,  
alpha=opacity,  
color='c',  
# edgecolor='r',
# hatch='xx', 
label='DExpert') 

  
plt.xlabel('Semantic Similarity', fontsize=12, fontweight='bold')  
plt.ylabel('Toxicity', fontsize=12, fontweight='bold')  
# plt.title('Toxicity vs Semantic Similarity')  
plt.xticks(index + bar_width * 1.5, ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], fontsize=12)
plt.yticks(fontsize=12) 
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray')

plt.legend(fontsize=10, loc='upper left')  

plt.tight_layout()  
plt.savefig("save_figs/half_toxic_similar_all_wo_prompt.png")