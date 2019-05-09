import h5py
import tables
import pickle


def load_vocab(vocab_file, sent):
    vocab = {}
    with open(vocab_file, 'rb') as f:
        word = ""
        info = pickle.load(f)  # dict
        for word, index in dict.items(info):
            vocab[index] = word
        for num in sent:
            word += vocab[num]
            word += " "
    return word


def process(rfile, wfile, vfile):
    table = tables.open_file(rfile)
    data, index = (table.get_node('/phrases'), table.get_node('/indices'))

    with open(wfile, "w+") as w:
        for offset in range(index.shape[0]):
            len, pos = index[offset]['length'], index[offset]['pos']
            sent = str(data[pos: pos + len].astype('int32')).replace("\n", " ")
            w.write(sent + '\n')
            w.flush()
            print(str(offset) + " " + sent)
            # word = load_vocab(vfile, sent)
            # print(word)
            # w.write(word + '\n')


feat = ["apiseq", "desc", "methname", "tokens"]

for name in feat:
    rfile = 'deepcs_dataset/test.' + name + '.h5'
    wfile = 'deepcs_dataset/test.' + name + '.txt'
    vfile = 'deepcs_dataset/vocab.' + name + '.pkl'
    process(rfile, wfile, vfile)
