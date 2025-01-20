import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

loads = []
lasts = []
deltas = []
rw_deltas = []

# 读取数据
with open("./gem5output_rv/load_log", "r") as file:
    print("[LOAD_PLT] plt doing...")
    for line in file:
        parts = line.split()
        load_value = int(parts[0].split(":")[1])
        last_value = int(parts[2].split(":")[1])
        delta_value = int(parts[3].split("=")[1])
        rw_delta_value = int(parts[4].split("=")[1])

        loads.append(load_value)
        lasts.append(last_value)
        deltas.append(delta_value)
        rw_deltas.append(rw_delta_value)

# 定义分组区间和标签
bins = [-float("inf"), 0, 16, 32, 64, 128, 256, float("inf")]
labels = ["<0", "0-16", "16-32", "32-64", "64-128", "128-256", "256-inf"]

# 将 deltas 分组
delta_groups = pd.cut(deltas, bins=bins, labels=labels, right=False)
group_counts = delta_groups.value_counts().sort_index()

# 将 rw_deltas 分组
rw_delta_groups = pd.cut(rw_deltas, bins=bins, labels=labels, right=False)
rw_group_counts = rw_delta_groups.value_counts().sort_index()

# 绘制所有图在一个图中
plt.figure(figsize=(20, 10), dpi=400)

# 图 1: Load vs Last
plt.subplot(2, 2, 1)
plt.plot(loads, lasts, marker="o", color="b", label="Load vs Last")
plt.xlabel("Load")
plt.ylabel("Last")
plt.title("Load vs Last")
plt.grid(True)
plt.legend()

# # 图 2: Load vs Delta
# plt.subplot(2, 3, 2)
# plt.plot(loads, deltas, marker="o", color="r", label="Load vs Delta")
# plt.xlabel("Load")
# plt.ylabel("Delta")
# plt.title("Load vs Delta")
# plt.grid(True)
# plt.legend()

# 图 3: Load vs rw_Delta
plt.subplot(2, 2, 2)
plt.plot(loads, rw_deltas, marker="o", color="g", label="Load vs rw_Delta")
plt.xlabel("Load")
plt.ylabel("rw_Delta")
plt.title("Load vs rw_Delta")
plt.grid(True)
plt.legend()

# # 图 4: Delta Value Distribution
# plt.subplot(2, 3, 4)
# group_counts.plot(kind="bar", color="skyblue", width=0.6)
# plt.xlabel("Delta Value Groups")
# plt.ylabel("Frequency")
# plt.title("Delta Value Distribution")
# plt.xticks(rotation=45)
# plt.grid(axis="y")

# 图 5: rw_Delta Value Distribution
plt.subplot(2, 1, 2)
rw_group_counts.plot(kind="bar", color="orange", width=0.6)
plt.xlabel("rw_Delta Value Groups")
plt.ylabel("Frequency")
plt.title("rw_Delta Value Distribution")
plt.xticks(rotation=45)
plt.grid(axis="y")

# 调整布局和保存
plt.tight_layout()
plt.savefig("./all_plots_in_one.png")

print("[LOAD_PLT]Plotting completed.")
