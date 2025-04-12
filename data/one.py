def process_and_write(input_file, output_file, chunk_size=30):
    with open(input_file, 'r') as f:
        numbers = list(map(int, f.read().split()))

    # Apply zero-indexing
    numbers = [n - 1 for n in numbers]

    # Break into chunks
    chunks = [numbers[i:i + chunk_size] for i in range(0, len(numbers), chunk_size)]

    # Remove chunks shorter than (window_size + 1)
    window_size = 10
    filtered_chunks = [chunk for chunk in chunks if len(chunk) > window_size]

    with open(output_file, 'w') as f:
        for chunk in filtered_chunks:
            f.write(' '.join(map(str, chunk)) + '\n')

    print(f"[+] Wrote {len(filtered_chunks)} sequences to {output_file}")


if __name__ == "__main__":
    process_and_write('normal_ids.txt', 'hdfs_test_normal')
    process_and_write('abnormal_ids.txt', 'hdfs_test_abnormal')
