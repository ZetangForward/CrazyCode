import matplotlib.pyplot as plt  
  
# 数据  
categories = ['Track1-D', 'Track2-D', 'Track1-F', 'Track2-F', 'Track1-R', 'Track2-R']  
win = [41.11, 37.4, 47.11, 43.7, 9.33, 10.56]  
tie = [50.45, 40.75, 11.33, 14.82, 81.33, 84.44]  
loss = [8.44, 21.85, 41.56, 41.48, 9.33, 5.0]  
  
# 创建条形图  
fig, ax = plt.subplots(figsize=(12, 6))  
  
bar_width = 0.7  # 设置条形图的宽度  
bar_positions = [i for i in range(len(categories))]  # 设置条形图的位置  
  
bars_win = ax.barh(bar_positions, win, height=bar_width, color='green', edgecolor='white', label='Win')  
bars_tie = ax.barh(bar_positions, tie, height=bar_width, left=win, color='blue', edgecolor='white', label='Tie')  
bars_loss = ax.barh(bar_positions, loss, height=bar_width, left=[i+j for i,j in zip(win, tie)], color='red', edgecolor='white', label='Loss')  
  
# 添加文本标签的函数  
def add_labels(bars):  
    for bar in bars:  
        width = bar.get_width()  
        label_x_pos = bar.get_x() + width / 2  
        label_y_pos = bar.get_y() + bar.get_height() / 2  
        ax.text(label_x_pos, label_y_pos, f'{width:.2f}%', ha='center', va='center', color='white',   
                fontsize=18, fontname='DejaVu Sans Mono', fontweight='bold', clip_on=False)  
  
# 添加文本标签  
add_labels(bars_win)  
add_labels(bars_tie)  
add_labels(bars_loss)  
  
# 设置图表标题和标签  
ax.set_yticks(bar_positions)  
ax.set_yticklabels(categories, fontsize=18, fontname='DejaVu Sans Mono', fontweight='bold')  
ax.invert_yaxis()  # 标签从上到下  
  
# 设置x轴的限制以确保最右边的标签可以贴在边框上  
ax.set_xlim(0, 100)  
  
# 设置x轴和y轴的标签大小  
ax.tick_params(axis='x', labelsize=18)  
ax.tick_params(axis='y', labelsize=18)  
  
# 将x轴的刻度标签设置为粗体  
for label in ax.get_xticklabels():  
    label.set_fontweight('bold')  
  
# 添加图例  
ax.legend(ncol=len(categories), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small', prop={'size': 18, 'weight': 'bold'})   


# 显示图表  
plt.tight_layout()  
plt.savefig('save_figs/human_eval.png', bbox_inches='tight')

