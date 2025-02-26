import re

def group_rr_data(data):
    """
    处理包含 "rr" 的数据行，并根据 "rr" 后的第一个数字（0~32）进行分组。
    
    参数：
        data (str)：包含多行数据的字符串。
    
    返回：
        dict：分组结果，键为第一个数字（0~32），值为对应的第二个参数列表。
    """
    # 定义正则表达式，匹配 "rr 数字 数字/十六进制"
    pattern = re.compile(r'rr\s+(\d+)\s+(0x[\da-fA-F]+|\d+)')
    
    # 初始化分组字典，键为第一个数字（0~32），值为列表
    groups = {str(i): [] for i in range(33)}  # 0~32
    
    # 按行处理数据
    for line in data.splitlines():
        # 查找当前行中的所有 "rr" 模式
        matches = pattern.findall(line)
        for match in matches:
            key = match[0]  # 第一个数字
            value = match[1]  # 第二个参数
            if key in groups:
                groups[key].append(value)
    
    return groups


if __name__ == "__main__":
    with open("./gem5output_rv/reg_dealer_output", "r") as file:
        data = file.read()
    # 调用函数
    result = group_rr_data(data)


    with open("./gem5output_rv/reg_dealer_output_classified", "w") as output:
        for key in range(33):
            if result[str(key)]:
                output.write(f"Group {key}: {result[str(key)]}\n")  # 打印每个组的前5个元素