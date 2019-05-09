import pickle


def geneVocabTXT(f, w):
    with open(f, 'rb') as f, open(w, 'w+') as w:
        info = pickle.load(f)  # dict

        for word, index in dict.items(info):
            w.write(word + '\n')
            w.flush()


pkl = ['ast', 'apiseq', 'tokens', 'desc', 'methname']
for name in pkl:
    f = 'data/test/pkl/test.' + name + '.pkl'
    w = 'data/test/vocab/' + name + '.txt'
    geneVocabTXT(f, w)
