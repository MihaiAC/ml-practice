import numpy as np

f = open("C:/Users/alexm/Desktop/word_alignments/europarl-v7.sv-en.en", 'r', encoding='utf8')
en_lines = f.readlines()
f.close()

g = open("C:/Users/alexm/Desktop/word_alignments/europarl-v7.sv-en.sv", 'r', encoding='utf8')
sv_lines = g.readlines()
g.close()

print(len(en_lines))
print(len(sv_lines))

random_ints = np.random.randint(0, len(en_lines), 10)
for idx in random_ints:
    print(en_lines[idx])
    print(sv_lines[idx])

