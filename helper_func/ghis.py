def process_file(input_file):
    buffer = []
    with open(input_file, "r") as infile:
        for line in infile:
            buffer.append(line[0])
    return buffer


def get_ghis(ghis, idx, length):
    start_idx = max(0, idx - length)
    extracted = ghis[start_idx:idx]
    extracted_str = "".join(extracted)
    padding = "0" * (length - len(extracted_str))
    return padding + extracted_str


if __name__ == "__main__":
    input_file = "../gem5output_rv/debug_log_dealed"
    output_file = "./ghis_log"
    print("[GHIS] file processing")
    ghis = process_file(input_file)

    # idx = 30
    # length = 30
    # result = get_ghis(ghis, idx, length)
    # print(f"Extracted bits: {result}")

    print(f"[GHIS] Processed data written to {output_file}")
