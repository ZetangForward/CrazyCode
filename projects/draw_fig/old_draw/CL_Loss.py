import matplotlib.pyplot as plt  
import numpy as np  

# golbal config
plt.rcParams['font.weight'] = 'bold'
bar_width = 0.2
opacity = 0.7
# fig, ax = plt.subplots(figsize=(8,6))  


similarity = ["GPT2-XL", "Flan T5 XL", "Alpaca","Llama2-7b","Llama2-13b"]  

# llama before mask
similarity_portion1_b = [0.71,1.18,0.91,1.51,1.29]  
toxicity_wo_gen1_b = [0.0824, 0.0948, 0.1375, 0.1691, 0.2033, 0.2371, 0.2783, 0.3381, 0.4081, 0.5277]  

# gpt2-xl before mask
similarity_portion2_b = [0.32,0.20,0.34,0.31,0.40]  
toxicity_wo_gen2_b = [0.1083,0.1161,0.1372,0.1668,0.2055,0.2371,0.2839,0.3323,0.4357,0.5252]

# llama after mask
# similarity_portion1 = [0.0249,0.0579,0.1021,0.1473,0.1742]  
similarity_portion1 = [31.87,42.83,44.73,47.88,48.34]  
toxicity_wo_gen1 = [0.0518,0.0593,0.0846,0.1048,0.1257,0.1474,0.181,0.1957,0.1937,0.2571]  

# gpt2-xl after mask
# similarity_portion2 = [0.0035,0.0222,0.0597,0.1143,0.1696]  
similarity_portion2 = [30.38,37.04,42.63,46.07,48.04]  
toxicity_wo_gen2 = [0.1066,0.1158,0.1137,0.1155,0.1268,0.1501,0.186,0.2235,0.2502,0.287]


index1 = np.arange(len(similarity_portion1))
  
fig, ax1 = plt.subplots()  
ax2 = ax1.twinx() 
  
bar1 = ax1.bar(index1 - bar_width, similarity_portion1_b, bar_width, alpha=opacity, color='b', label='Toxicity Prob. w/o CL')  
bar2 = ax1.bar(index1 , similarity_portion2_b, bar_width, alpha=opacity, color='r', label='Toxicity Prob. w CL')  
bar1 = ax2.bar(index1 + bar_width*1, similarity_portion1, bar_width, alpha=opacity, color='none', edgecolor='b', hatch='xx', label='PPL w/o CL')  
bar2 = ax2.bar(index1 + bar_width*2, similarity_portion2, bar_width, alpha=opacity, color='none', edgecolor='r', hatch='xx', label='PPL w CL') 
 

# bar3 = plt.bar(index1 + 2*bar_width, similarity_portion3, bar_width, alpha=opacity, color='g', label='Gedi')  
# bar4 = plt.bar(index1 + 3*bar_width, similarity_portion4, bar_width, alpha=opacity, color='c', label='DExpert')  

# bar3 = plt.bar(index1 + 2*bar_width, similarity_portion3, bar_width, alpha=opacity, color='none', edgecolor='b', hatch='xx', label='Gedi')  
# bar4 = plt.bar(index1 + 3*bar_width, similarity_portion4, bar_width, alpha=opacity, color='none', edgecolor='r', hatch='xx', label='Dexpert')  
    

ax1.set_ylabel('Toxicity Prob.', fontsize=11, fontweight='bold')  
plt.xticks([i + bar_width / 2 for i in index1], ["GPT2-XL", "Flan-T5-XL", "Alpaca","LLaMA2-7b","LLaMA2-13b"] , fontsize=10)  
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray')
ax1.legend(fontsize=10, loc='upper left')  
ax1.set_ylim([0, 2])


ax2.set_ylabel('PPL', fontsize=10, fontweight='bold')  
ax2.set_ylim([0, 70])
plt.yticks(fontsize=11)  
ax2.legend(fontsize=10, loc='upper right')


# ax2 = ax1.twinx()  
# ax2.set_ylabel('Toxicity')  
  
# # Plot toxicity_wo_gen1 and toxicity_wo_gen2 as line plots  
# ax2.plot(index1, toxicity_wo_gen1, color='blue', marker='o', label='Toxicity Llama')  
# ax2.plot(index2, toxicity_wo_gen2, color='red', marker='o', label='Toxicity GPT2-xl')  
# ax2.legend(loc='upper right')  
  
plt.tight_layout() 
plt.savefig("save_figs/with_cl.pdf") 