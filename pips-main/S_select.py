import matplotlib.pyplot as plt

# 数据
S = [4, 8, 16]
time = [0.89, 0.85, 0.67]
acc = [0.7119, 0.5999, 0.4579]

# 绘图
fig, ax = plt.subplots()

bar_width = 0.35
index = [i for i in range(len(S))]

rects1 = ax.bar(index, time, bar_width, label='Time')
rects2 = ax.bar([i + bar_width for i in index], acc, bar_width, label='Accuracy')

ax.set_xlabel('S')
ax.set_ylabel('Value')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(S)
ax.legend()

# 先保存PDF，再显示图形
plt.savefig('Time_Accuracy_vs_S.pdf')
plt.show()