import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 读取文本文件，解析数据
def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # 每行由两个数字组成，用空格分隔
            data.append([int(x) for x in line.strip().split()])
    return np.array(data)


# 将数据整理成长度为1024的历史数据
def create_sequences(data, seq_length=1024):
    mask = 196
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - mask):
        sequence = data[i : i + seq_length].copy()  # 获取1024长度的序列
        extra_feature = data[i + seq_length + mask, 0]  # 附加的特征值
        sequence = np.append(sequence, [[extra_feature, 0]], axis=0)  # 添加到序列末尾
        sequences.append(sequence.flatten())  # 展平序列以供 LightGBM 使用
        labels.append(data[i + seq_length + mask, 1])  # 标签为下一个时间步目标值
    return np.array(sequences), np.array(labels)


# 重新映射标签到 [0, num_classes-1]
def remap_labels(labels):
    unique_labels = np.unique(labels)  # 找出唯一的标签值
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    remapped_labels = np.array([label_mapping[label] for label in labels])
    return remapped_labels, len(unique_labels)  # 返回重映射后的标签和类别数


# 数据标准化（仅对输入特征，标签保持整数）
def standardize_data(sequences):
    scaler = StandardScaler()
    sequences = scaler.fit_transform(sequences)  # 对特征进行标准化
    return sequences


# 主函数
def main():
    # 读取数据
    file_path = "./gem5output_rv/model_data"  # 假设文本文件路径为 'data.txt'
    data = load_data(file_path)

    # 创建长度为1024的历史序列
    seq_length = 1024
    sequences, labels = create_sequences(data, seq_length)

    # 重新映射标签
    labels, num_classes = remap_labels(labels)
    print(f"Number of classes: {num_classes}")

    # 标准化数据
    sequences = standardize_data(sequences)

    # 划分训练集和验证集（80%训练，20%验证）
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    # 转换为 LightGBM 数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # LightGBM 参数
    params = {
        "objective": "multiclass",  # 多分类任务
        "num_class": num_classes,  # 类别数
        "boosting_type": "gbdt",  # 梯度提升决策树
        "metric": "multi_logloss",  # 多分类的损失函数
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "seed": 42,
    }

    # 训练 LightGBM 模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,  # 提升树的数量
        valid_sets=[train_data, val_data],  # 验证集
    )

    # 预测与评估
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_val, y_pred_classes)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
