def extract_numbers(file_path, output_file):
    with open(file_path, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.split()
            second_number = parts[1]
            rw_delta_value = next(
                part.split("=")[1] for part in parts if part.startswith("rw_delta=")
            )
            idx = parts[0]
            idx = idx.split(":")[1]
            outfile.write(
                str(idx) + " " + str(second_number) + " " + str(rw_delta_value) + "\n"
            )


file_path = "./load_log"
output_file = "./model_data"
extract_numbers(file_path, output_file)
print(f"[MODEL_DATA] Processed data written to {output_file}")
