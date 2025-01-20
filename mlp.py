import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
        sequences.append(sequence)
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
    # 只对输入特征进行标准化
    features = sequences[:, :, 0]  # 提取第一个特征
    features = scaler.fit_transform(features)
    sequences[:, :, 0] = features  # 替换标准化后的特征
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

    # 展平输入数据以适应 MLP
    X = sequences[:, :, 0].reshape(sequences.shape[0], -1)

    # 划分训练集和验证集（80%训练，20%验证）
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # 构建并训练 MLP 模型
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),  # 两个隐藏层，分别有256和128个神经元
        activation="relu",  # 激活函数使用 ReLU
        solver="adam",  # 优化器使用 Adam
        max_iter=200,  # 最大迭代次数
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 验证模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # 测试模型 - 选取10组样本进行预测
    test_samples = 10
    test_indices = np.random.choice(len(X_val), test_samples, replace=False)
    test_data = X_val[test_indices]
    actual_values = y_val[test_indices]

    predicted_values = model.predict(test_data)
    print("Predicted:", predicted_values)
    print("Actual:", actual_values)


if __name__ == "__main__":
    main()
