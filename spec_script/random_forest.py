import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(file_path, ghis_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append([int(x) for x in line.strip().split()])
    ghis = []
    with open(ghis_path, "r") as file:
        for line in file:
            ghis.append([int(x) for x in line.strip().split()])
    return np.array(data), np.array(ghis)


def create_sequences(data, ghis, seq_length=1024):
    mask = 128
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - mask - 1):
        load_idx = data[i + seq_length + mask, 0]  # 希望预测的load的idx
        # if i == 0:
        # print("idx=", load_idx)
        if load_idx < seq_length:
            continue
        sequence = data[i : i + seq_length, 1:3].copy()  # 获取1024长度的序列
        ghis_data = ghis[load_idx - seq_length : load_idx, 0:2].copy()
        sequence = np.concatenate(
            (sequence, ghis_data), axis=0
        )  # 沿着axis=0拼接（行拼接）
        extra_feature = data[i + seq_length + mask, 0]  # 附加的特征值
        extra_feature = np.array([[extra_feature, extra_feature]])
        sequence = np.concatenate(
            (sequence, extra_feature), axis=0
        )  # 额外特征添加为二维数组
        if sequence.shape != (2049, 2):
            print(f"Error: sequence shape is {sequence.shape}, expected (2049, 2).")
            print("i=", i)
            print("load_idx=", load_idx)
            exit("Exiting program due to incorrect sequence shape.")
        sequences.append(sequence)
        labels.append(data[i + seq_length + mask, 2])  # 标签为下一个时间步目标值
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


def main():
    print("[RANDOM_FOREST BEGIN]")
    file_path = "./model_data"
    ghis_path = "./ghis_log"
    data, ghis = load_data(file_path, ghis_path)

    # 创建长度为1024的历史序列
    seq_length = 1024
    sequences, labels = create_sequences(data, ghis, seq_length)

    # 重新映射标签
    labels, num_classes = remap_labels(labels)
    print(f"Number of classes: {num_classes}")

    # 标准化数据
    sequences = standardize_data(sequences)

    # 展平输入数据以适应随机森林
    X = sequences[:, :, 0].reshape(sequences.shape[0], -1)

    # 划分训练集和验证集（80%训练，20%验证）
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # 构建并训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    print("[RANDOM_FOREST DONE]")


if __name__ == "__main__":
    main()
