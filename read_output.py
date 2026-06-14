with open("d:/Worksolo/s0003/train_small_output.txt", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"{i+1}: {line.rstrip()}")
