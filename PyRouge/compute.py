import sys

from Rouge import Rouge

rouge = Rouge.Rouge()

ref_file = sys.argv[1]
system_file = sys.argv[2]

systems = []
refs = []

with open(system_file, encoding='utf-8') as f:
    for line in f:
        if not line:
            break

        systems.append(line.strip())

with open(ref_file, encoding='utf-8') as f:
    for line in f:
        if not line:
            break
        refs.append(line.strip())

scores = rouge.compute_rouge(refs, systems)
print(scores)
