import re


def is_hex(value):
    return bool(re.match(r"^0x[0-9a-fA-F]+$", value))


def process_file(input_file_path, output_file_path):
    with (
        open(input_file_path, "r") as input_file,
        open(output_file_path, "w") as output_file,
    ):
        lines = input_file.readlines()
        for index in range(len(lines) - 1):
            parts = lines[index].strip().split(" ")
            if len(parts) >= 2:
                bool_value = parts[0]
                hex_value = parts[1]
                next_parts = lines[index + 1].strip().split(" ")
                if is_hex(hex_value) != 1:
                    continue
                if len(next_parts) >= 2:
                    next_hex_value = next_parts[1]
                    if is_hex(next_hex_value) != 1:
                        continue
                    indicator = 0
                    if "Call" in lines[index]:
                        indicator = 1
                    elif "Return" in lines[index]:
                        indicator = 2
                    elif "Indirect" in lines[index]:
                        indicator = 3
                    output_line = (
                        f"{bool_value} {hex_value} {next_hex_value} {indicator}\n"
                    )
                    output_file.write(output_line)


input_file_path = "./gem5output_rv/debug_log_dealed"
output_file_path = "./gem5output_rv/fronted_log"
process_file(input_file_path, output_file_path)
print(f"[FRONTED_TRACE] Processed data written to {output_file_path}")
