import pickle
import h5py
import configs

cf = configs.conf()

def parse_input(text):
    tok = [z for z in text.split(' ')]
    return [z for z in tok]


def preprocess(file):
    counts = {}
    vocab = set()
    sent = []
    with open(file, 'r') as f:
        for line in f.readlines():
            words = parse_input(line.rstrip())
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            sent.append(words)

            vocab.update(words)

    return sent, vocab, counts


UNK = "<UNK>"


def prepare_vocab(vocab, counts, dict_size, file):
    vocab2int = {}
    int2vocab = {}

    _sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
    _sorted = [UNK] + _sorted
    for i, word in enumerate(_sorted):
        if i > dict_size:
            break
        vocab2int[word] = i
        int2vocab[i] = word

    # 词表写入pkl文件(dict)
    with open(file, 'wb') as f:
        pickle.dump(vocab2int, f, pickle.HIGHEST_PROTOCOL)
    return vocab2int, int2vocab


def words2ids(text, vocab2int, file):
    # 训练数据转成数字表示
    with open(file, 'w+') as f:
        for line in text:
            r = ([vocab2int.get(x, vocab2int[UNK]) for x in line])
            for num in r:
                f.write(str(num) + ' ')
            f.write('\n')


def load_data(raw_file, h5_file, pkl_file):
    sent, vocab, counts = preprocess(raw_file)
    vocab2int, int2vocab = prepare_vocab(vocab, counts, cf.n_words - 1, pkl_file)
    words2ids(sent, vocab2int, h5_file)


if __name__ == '__main__':
    features = ['desc', 'tokens', 'apiseq', 'methname']
    for feat in features:
        raw_feat = 'data/v1/train/original/v1.' + feat + '.txt'
        enc_feat = 'data/v1/train/enc/v1.' + feat + '.enc'
        pkl_feat = 'data/v1/vocab/v1.' + feat + '.pkl'
        load_data(raw_feat, enc_feat, pkl_feat)
