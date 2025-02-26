import re

def extract_rr_rw_entries(line):
    # pattern = re.compile(r'\b(r[wr])\s+(\d+|0x[\da-fA-F]+)\s+(\d+|0x[\da-fA-F]+)\b')
    pattern = re.compile(r'\b(r[r])\s+(\d+|0x[\da-fA-F]+)\s+(\d+|0x[\da-fA-F]+)\b')
    matches = pattern.findall(line)
    return ' '.join(item for match in matches for item in match)

# 示例用法
if __name__ == "__main__":
    log_file = "./gem5output_rv/simple_debug_log_dealed"
    output_file = "./gem5output_rv/reg_dealer_output"
    with open(log_file, "r") as file:
        with open(output_file, "w") as output:
            for line in file:
                result = extract_rr_rw_entries(line)
                # not empty
                if result:
                    output.write(f"{result}\n")
                    