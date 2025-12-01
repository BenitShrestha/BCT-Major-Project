def split_file_into_parts(filename, num_parts=6):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_lines = len(lines)
    part_size = total_lines // num_parts

    for i in range(num_parts):
        start = i * part_size
        # Last part takes the remainder
        end = (i + 1) * part_size if i < num_parts - 1 else total_lines

        part_lines = lines[start:end]
        part_filename = f"{filename}_part{i+1}.txt"

        with open(part_filename, 'w', encoding='utf-8') as part_file:
            part_file.writelines(part_lines)

        print(f"Written {len(part_lines)} lines to {part_filename}")

# Example usage
split_file_into_parts("nepberta_text.txt", num_parts=6)
