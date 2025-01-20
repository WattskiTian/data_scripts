import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard  # 引入 TensorBoard 回调


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
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])  # 获取1024长度的序列
        labels.append(data[i + seq_length, 1])  # 标签为下一个时间步目标值
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


# 构建RNN模型
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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

    # 创建 TensorBoard 回调，保存日志到指定目录
    log_dir = "./rnn_logs"  # 指定日志保存目录
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 构建并训练模型
    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    # 训练模型，并加入 TensorBoard 回调
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback],  # 加入 TensorBoard 回调
    )

    # # 测试模型 - 选取10组样本进行预测
    # test_samples = 10
    # test_indices = np.random.choice(len(X_val), test_samples, replace=False)
    # test_data = X_val[test_indices]
    # actual_values = y_val[test_indices]
    #
    # # 使用模型进行预测


if __name__ == "__main__":
    main()
