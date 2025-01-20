import re


def process_file(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        lines = infile.readlines()
        Coverage_cnt = 0
        load_cnt = 0
        for i, line in enumerate(lines):
            flag, ld_reg_num = get_load(line)
            if flag == 0:
                continue
            else:
                load_cnt += 1
                # end_idx = max(0, i - 1024)
                end_idx = 0
                reg_delta_max = 1024
                rw_delta = 0
                for j in range(i - 1, end_idx, -1):
                    if "rw" in lines[j]:
                        rw_delta += 1

                    if rw_delta > reg_delta_max:
                        break

                    pat = "rw " + ld_reg_num + " "
                    if pat in lines[j]:
                        delta = int(i) - int(j)
                        outfile.write(
                            "Load:"
                            + str(i)
                            + " "
                            + ld_reg_num
                            + " "
                            + "Last:"
                            + str(j)
                            + " "
                            + "delta="
                            + str(delta)
                            + " "
                            + "rw_delta="
                            + str(rw_delta)
                            + "\n"
                        )
                        Coverage_cnt += 1
                        break
        Coverage_ratio = Coverage_cnt / load_cnt
        print("reg delta max = 1024, Coverage = ", Coverage_ratio)


def get_load(line):
    match = re.search(r"\(Load\)\s+rr\s+(\d+)", line)
    if match:
        flag = 1
        ret = match.group(1)
        return flag, ret
    else:
        return 0, None


if __name__ == "__main__":
    input_file = "./gem5output_rv/debug_log_dealed"
    output_file = "./gem5output_rv/load_log"
    print("[LOAD] file processing")
    process_file(input_file, output_file)

    print(f"[LOAD] Processed data written to {output_file}")
