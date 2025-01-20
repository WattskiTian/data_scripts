def extract_first_characters(input_file, output_file):
    # 打开输入文件读取
    with open(input_file, "r") as infile:
        concatenated_characters = "\n".join(
            line[0] + " " + line[0] for line in infile if line.strip()
        )  # 忽略空行

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(concatenated_characters)


if __name__ == "__main__":
    input_file = "./gem5output_rv/debug_log_dealed"
    output_file = "./gem5output_rv/ghis_log"
    extract_first_characters(input_file, output_file)
