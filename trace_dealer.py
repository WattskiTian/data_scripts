import re
import gzip


def process_line(line):
    # Case 1: Format: system.cpu.[tid:0]: Setting integer register integer[3] (3) to 0x7e474.
    match1 = re.match(
        r"system\.cpu\.\[tid:\d+\]: Setting integer register integer\[(\d+)\] \(\d+\) to ((?:0x)?[0-9a-fA-F]+)\.",
        line,
    )
    if match1:
        ret = f"rw {match1.group(1)} {match1.group(2)}"
        flag = 0
        return flag, ret
    # Case 2: Format: system.cpu.[tid:0]: Reading integer reg integer[15] (15) as 0x7ae90.
    match2 = re.match(
        r"system\.cpu\.\[tid:\d+\]: Reading integer reg integer\[(\d+)\] \(\d+\) as ((?:0x)?[0-9a-fA-F]+)\.",
        line,
    )
    if match2:
        ret = f"rr {match2.group(1)} {match2.group(2)}"
        flag = 0
        return flag, ret
    match3 = re.match(r".*:\s*(0x[0-9a-fA-F]+).*?:.*?:\s*(.*)", line)
    if match3:
        # Extract group2 (the instruction details) and apply the replacements
        group2 = match3.group(2)
        group2 = group2.replace("IntAlu", "A")  # Replace IntAlu with A
        group2 = group2.replace("MemWrite", "Mw")  # Replace MemWrite with Mw
        group2 = group2.replace("MemRead", "Mr")
        group2 = group2.replace("flags=", "")  # Remove flags=
        group2 = group2.replace("Is", "")  # Remove Is
        group2 = group2.replace("Integer", "")  # Remove Integer
        group2 = group2.replace("()", "")  # Remove ()
        group2 = group2.replace("  ", " ")
        group2 = group2.replace("(|", "(")
        ret = f"{match3.group(1)} {group2}"
        flag = 1
        return flag, ret
    match4 = re.match(r"system\.cpu: Brach_Taken : (\d+)", line)
    if match4:
        ret = f"{match4.group(1)}"
        flag = 2
        return flag, ret
    print("no match with : ", line)
    flag = 3
    return flag, line


def process_file(input_file, output_file):
    buffer = []
    first_line = 1
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if first_line == 1:
                first_line = 0
                continue
            flag, processed_line = process_line(line)
            if flag == 0:
                buffer.append(processed_line)
            elif flag == 1:
                buffer.insert(0, processed_line)
                # outfile.write(" ".join(buffer) + "\n")
                # buffer.clear()
            elif flag == 2:
                buffer.insert(0, processed_line)
                if len(buffer) > 1:
                    outfile.write(" ".join(buffer) + "\n")
                buffer.clear()
            else:
                continue


def process_gz_file(input_file, output_file):
    buffer = []
    first_line = 1
    with gzip.open(input_file, "rt") as infile, gzip.open(output_file, "wt") as outfile:
        for line in infile:
            if first_line == 1:
                first_line = 0
                continue
            flag, processed_line = process_line(line.strip())
            if flag == 0:
                buffer.append(processed_line)
            else:
                buffer.insert(0, processed_line)
                outfile.write(" ".join(buffer) + "\n")
                buffer.clear()


if __name__ == "__main__":
    input_file = "./gem5output_rv/debug_log"
    output_file = "./gem5output_rv/debug_log_dealed"

    if input_file.endswith(".gz"):
        print("[TRACE_DEALER] gz file processing")
        output_file = "./gem5output_rv/debug_log_dealed.gz"
        process_gz_file(input_file, output_file)
    else:
        print("[TRACE_DEALER] file processing")
        output_file = "./gem5output_rv/debug_log_dealed"
        process_file(input_file, output_file)

    print(f"[TRACE_DEALER] Processed data written to {output_file}")
