# read data
import openpyxl  
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

# golbal config
plt.rcParams['font.weight'] = 'bold'
bar_width = 0.3
opacity = 0.7
fig, ax = plt.subplots(figsize=(8,6))  

# 加载工作簿（将路径替换为您的文件路径）  
workbook = openpyxl.load_workbook('data.xlsx')  
  
# 获取工作表（这将获取第一个工作表，如果有多个，请根据需要修改）  
worksheet = workbook.active  
  
# 创建一个空列表以保存数据  
data = []  
  
# 通过行和列迭代数据并保存到列表中  
for row in worksheet.iter_rows(min_row=1, max_row=11, min_col=5, max_col=11):  
    row_data = []  
    for cell in row:  
        row_data.append(cell.value)  
    data.append(row_data)  
  
# 关闭工作簿  
workbook.close()  

toxic_portation = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
str_toxic_portation = data[0]
prompt_portion = data[1]
prompt_toxicity = data[2]
llama_toxicity = data[3]
gptxl_toxicity = data[4]
dexperts_toxicity = data[5]
gedi_toxicity = data[6]

mask_llama_toxicity = data[7]
mask_gptxl_toxicity = data[8]
mask_dexperts_toxicity = data[9]
mask_gedi_toxicity = data[10]

# print(llama_toxicity)

index = np.arange(len(toxic_portation))  

# 绘制直方图  
# plt.bar(index, llama_toxicity, bar_width, label='LLaMA-7B', alpha=opacity, color='b')  
# plt.bar(index + bar_width, gptxl_toxicity, bar_width, label='GPT2-XL', alpha=opacity, color='r')  
bar1 = plt.plot(index, llama_toxicity, linewidth=2, marker='o', linestyle='--', label='LLaMA2-7B', alpha=opacity, color='b')  
bar2 = plt.plot(index + bar_width, gptxl_toxicity, linewidth=2, marker='o', linestyle='--', label='GPT2-XL', alpha=opacity, color='r')  

# plt.bar(index + bar_width * 2, gedi_toxicity, bar_width, label='Gedi', alpha=opacity, color='g')  
# plt.bar(index + bar_width * 3, dexperts_toxicity, bar_width, label='DExpert', alpha=opacity, color='c')  

# bar1 = plt.bar(index + bar_width*2, mask_llama_toxicity, bar_width, alpha=opacity, color='none', edgecolor='b', hatch='xx', label='LLaMA-7B (mask)')  
# bar2 = plt.bar(index + bar_width*3, mask_gptxl_toxicity, bar_width, alpha=opacity, color='none', edgecolor='r', hatch='xx', label='GPT2-XL (mask)') 


# 添加标题和坐标轴标签  
# plt.title('Toxicity Comparison')  

# y_major_locator=MultipleLocator(0.5)
# ax.yaxis.set_major_locator(y_major_locator)

plt.xlabel('Prompt Toxicity', fontsize=17, fontweight='bold')  
plt.ylabel('Generation Toxicity', fontsize=17, fontweight='bold')   
plt.ylim(0, 0.6)
plt.xticks(index + 0.5*bar_width, str_toxic_portation, fontsize=14)  
plt.yticks(fontsize=17) 
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray')
plt.legend(fontsize=14, loc='upper left')  

ax2 = ax.twinx()  
ax2.set_ylabel('Generation Toxicity (mask)', fontsize=17, fontweight='bold')  
plt.ylim(0, 0.6)
plt.yticks(fontsize=17)  
# bar1 = plt.plot(index, mask_llama_toxicity, linewidth=2, marker='o', linestyle='--', alpha=opacity, color='b', label='LLaMA-7B (mask)')  
# bar2 = plt.plot(index + bar_width, mask_gptxl_toxicity, linewidth=2, marker='o', linestyle='--', alpha=opacity, color='r', label='GPT2-XL (mask)') 
plt.bar(index, mask_llama_toxicity, bar_width, alpha=opacity, color='b', label='LLaMA2-7B (mask)')  
plt.bar(index + bar_width, mask_gptxl_toxicity, bar_width, alpha=opacity, color='r', label='GPT2-XL (mask)') 
plt.legend(fontsize=14, loc='upper right')  

plt.tight_layout()     
plt.savefig(r"save_figs/mask_toxic_prompt_generation_backbone_test")
# 显示图形
# plt.show()



