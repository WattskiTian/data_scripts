import re

def process_line(line):
    hex_match = re.search(r"0x[0-9a-fA-F]+", line)
    hex_part = hex_match.group(0)
    bin_part = bin(int(hex_part, 16))[2:].zfill(20)[:-1]
    rr_count = line.count("rr")
    if rr_count == 0:
        bit_rr = "00"
    elif rr_count == 1:
        bit_rr = "10"
    elif rr_count >= 2:
        bit_rr = "11"

    bit_rw = "1" if "rw" in line else "0"
    bit_Store = "1" if "Store" in line else "0"
    bit_Load = "1" if "Load" in line else "0"
    bit_Call = "1" if "Call" in line else "0"
    bit_Return = "1" if "Return" in line else "0"
    bit_CondControl = "1" if "CondControl" in line else "0"
    bit_UncondControl = "1" if "UncondControl" in line else "0"

    # if bit_UncondControl=="1":
    #     bit_branch_taken="1"
    # elif 

    return f"{bin_part} {bit_rr}{bit_rw}{bit_Store}{bit_Load}{bit_Call}{bit_Return}{bit_CondControl}{bit_UncondControl}"

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            processed_line = process_line(line.strip())
            outfile.write(processed_line + '\n')

if __name__ == "__main__":
    input_file = './gem5output_rv/debug_log_dealed'
    output_file = './gem5output_rv/data_dealed'
    process_file(input_file, output_file)
    print(f"[DATA_DEALER] Processing complete. Results saved in {output_file}")
