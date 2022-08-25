import matchzoo as mz
from tqdm import tqdm

preprocessor = mz.load_preprocessor('./full_data_results/basic_data_full/save/preprocessor')

vocab = preprocessor._context['vocab_unit']._context['term_index'].keys()

print(len(vocab))

in_file = '/data/users/caiyinqiong/.matchzoo/datasets/fasttext/wiki.en.vec'
out_file = '/data/users/caiyinqiong/.matchzoo/datasets/fasttext/wiki.qqp.en.vec'

out = open(out_file, 'w')
with open(in_file, 'r') as f:
    out.write(f.readline())
    for line in tqdm(f):
        current_word = line.rstrip().split(' ')[0]
        if current_word in vocab:
            out.write(line)
out.close()
