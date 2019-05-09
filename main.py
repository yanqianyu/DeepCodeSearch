import os
import pickle
import tables
import configs
import codecs
import numpy as np
import random
from scipy.stats import rankdata
import math
import traceback
import threading
from utils import normalize, cos_np_for_normalized, cos_np
import argparse
from models import *

class CodeSearcher:
    def __init__(self, conf):
        self.conf = conf
        self.path = self.conf.data_dir
        self.vocab_methname = self.load_pickle(self.path + self.conf.vocab_methname)
        self.vocab_apiseq = self.load_pickle(self.path + self.conf.vocab_apiseq)
        self.vocab_tokens = self.load_pickle(self.path + self.conf.vocab_tokens)
        self.vocab_desc = self.load_pickle(self.path + self.conf.vocab_desc)

        self.code_base = None
        self.code_base_chunksize = 2000000
        self.code_reprs = None
        self._eval_sets = None

    def load_pickle(self, filename):
        return pickle.load(open(filename, 'rb'))

    def load_hdf5(self, file, start_offset, chunk_size):
        table = tables.open_file(file)
        data, index = (table.get_node('/phrases'), table.get_node('/indices'))
        data_len = index.shape[0]
        if chunk_size == -1:  # load all data
            chunk_size = data_len
        start_offset = start_offset % data_len
        offset = start_offset
        sents = []
        while offset < start_offset + chunk_size:
            if offset >= data_len:
                chunk_size = start_offset + chunk_size - data_len
                start_offset = 0
                offset = 0
            len, pos = index[offset]['length'], index[offset]['pos']
            offset += 1
            sents.append(data[pos: pos + len].astype('int32'))
        table.close()
        return sents

    def load_txt_data(self, file, start_offset, chunk_size):
        with open(file, 'r') as f:
            lines = f.readlines()
            data_len = len(lines)
            if chunk_size == -1:  # load all data
                chunk_size = data_len
            start_offset = start_offset % data_len
            offset = start_offset
            sents = []
            while offset < start_offset + chunk_size:
                if offset >= data_len:
                    chunk_size = start_offset + chunk_size - data_len
                    start_offset = 0
                    offset = 0
                sent = [int(x, base=10) for x in lines[offset].rstrip().split(' ')]
                offset += 1
                sents.append(sent)
        return sents

    def load_train_data(self, offset, chunk_size):
        chunk_methnames = self.load_hdf5(self.path + self.conf.train_methodname, offset, chunk_size)
        chunk_apiseq = self.load_hdf5(self.path + self.conf.train_apiseq, offset, chunk_size)
        chunk_tokens = self.load_hdf5(self.path + self.conf.train_tokens, offset, chunk_size)
        chunk_descs = self.load_hdf5(self.path + self.conf.train_desc, offset, chunk_size)

        # chunk_methnames = self.load_txt_data(self.path + self.conf.train_methodname, offset, chunk_size)
        # chunk_apiseq = self.load_txt_data(self.path + self.conf.train_apiseq, offset, chunk_size)
        # chunk_tokens = self.load_txt_data(self.path + self.conf.train_tokens, offset, chunk_size)
        # chunk_descs = self.load_txt_data(self.path + self.conf.train_desc, offset, chunk_size)
        return chunk_methnames, chunk_apiseq, chunk_tokens, chunk_descs

    def load_valid_data(self, chunk_size):
        chunk_methnames = self.load_hdf5(self.path + self.conf.valid_methodname, 0, chunk_size)
        chunk_apiseq = self.load_hdf5(self.path + self.conf.valid_apiseq, 0, chunk_size)
        chunk_tokens = self.load_hdf5(self.path + self.conf.valid_tokens, 0, chunk_size)
        chunk_descs = self.load_hdf5(self.path + self.conf.valid_desc, 0, chunk_size)
        return chunk_methnames, chunk_apiseq, chunk_tokens, chunk_descs

    def load_use_data(self):
        methnames = self.load_hdf5(self.path + self.conf.use_methodname, 0, -1)
        apiseq = self.load_hdf5(self.path + self.conf.use_apiseq, 0, -1)
        tokens = self.load_hdf5(self.path + self.conf.use_tokens, 0, -1)
        return methnames, apiseq, tokens

    def load_codebase(self):
        if self.code_base == None:
            code_base = []
            codes = codecs.open(self.path + self.conf.use_codebase, encoding='utf-8', errors='replace').readlines()

            for i in range(0, len(codes), self.code_base_chunksize):
                code_base.append(codes[i: i + self.code_base_chunksize])
            self.code_base = code_base

    def load_code_reprs(self):
        if self.code_reprs == None:
            codereprs = []
            h5f = tables.open_file(self.path + self.conf.use_codevecs)
            vecs = h5f.root.vecs
            for i in range(0, len(vecs), self.code_base_chunksize):
                codereprs.append(vecs[i: i + self.code_base_chunksize])
            h5f.close()
            self.code_reprs = codereprs
        return self.code_reprs

    def save_code_reprs(self, vecs):
        npvecs = np.array(vecs)
        fvec = tables.open_file(self.path + self.conf.use_codevecs, 'w')
        atom = tables.Atom.from_dtype(npvecs.dtype)
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape, filters=filters)
        ds[:] = npvecs
        fvec.close()

    def convert(self, vocab, words):
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [vocab.get(w, 0) for w in words]

    def revert(self, vocab, indices):
        ivocab = dict((v, k) for k, v in vocab.items())
        return [ivocab.get(i, 'UNK') for i in indices]

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    def save_model_epoch(self, model, epoch):
        if not os.path.exists(self.path + 'models/' + self.conf.model_name + '/'):
            os.makedirs(self.path + 'models/' + self.conf.model_name + '/')
        model.save("{}models/{}/epo{:d}_code.h5".format(self.path, self.conf.model_name, epoch),
                   "{}models/{}/epo{:d}_desc.h5".format(self.path, self.conf.model_name, epoch),
                   overwrite=True)

    def load_model_epoch(self, model, epoch):
        assert os.path.exists(
            "{}models/{}/epo{:d}_code.h5".format(self.path, self.conf.model_name, epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        assert os.path.exists(
            "{}models/{}/epo{:d}_desc.h5".format(self.path, self.conf.model_name, epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        model.load("{}models/{}/epo{:d}_code.h5".format(self.path, self.conf.model_name, epoch),
                   "{}models/{}/epo{:d}_desc.h5".format(self.path, self.conf.model_name, epoch))

    def train(self, model):
        if self.conf.reload > 0:
            self.load_model_epoch(model, self.conf.reload)
        valid_every = self.conf.valid_every
        save_every = self.conf.save_every
        batch_size = self.conf.batch_size
        nb_epoch = self.conf.nb_epoch

        split = self.conf.validation_split
        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(self.conf.reload, nb_epoch):
            print('Epoch %d' % i, end=' ')
            chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_descs = self.load_train_data(
                i * self.conf.chunk_size, self.conf.chunk_size)

            chunk_padded_methnames = self.pad(chunk_methnames, self.conf.methname_len)
            chunk_padded_apiseqs = self.pad(chunk_apiseqs, self.conf.apiseq_len)
            chunk_padded_tokens = self.pad(chunk_tokens, self.conf.tokens_len)
            chunk_padded_good_descs = self.pad(chunk_descs, self.conf.desc_len)
            chunk_bad_descs = [desc for desc in chunk_descs]
            random.shuffle(chunk_bad_descs)
            chunk_padded_bad_descs = self.pad(chunk_bad_descs, self.conf.desc_len)

            hist = model.fit(
                [chunk_padded_methnames, chunk_padded_apiseqs, chunk_padded_tokens, chunk_padded_good_descs,
                 chunk_padded_bad_descs], epochs=1, batch_size=batch_size, validation_split=split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if valid_every is not None and i % valid_every == 0:
                acc1, mrr = self.valid(model, 1000, 1)

            if save_every is not None and i % save_every == 0:
                self.save_model_epoch(model, i)

    def valid(self, model, poolsize, K):
        #  poolsize - size of the code pool, if -1, load the whole test set
        if self._eval_sets is None:
            # self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
            methnames, apiseqs, tokens, descs = self.load_valid_data(poolsize)
            self._eval_sets = dict()
            self._eval_sets['methnames'] = methnames
            self._eval_sets['apiseqs'] = apiseqs
            self._eval_sets['tokens'] = tokens
            self._eval_sets['descs'] = descs

        c_1, c_2 = 0, 0
        data_len = len(self._eval_sets['descs'])
        for i in range(data_len):
            bad_descs = [desc for desc in self._eval_sets['descs']]
            random.shuffle(bad_descs)
            descs = bad_descs
            descs[0] = self._eval_sets['descs'][i]  # good desc
            descs = self.pad(descs, self.conf.desc_len)
            methnames = self.pad([self._eval_sets['methnames'][i]] * data_len, self.conf.methname_len)
            apiseqs = self.pad([self._eval_sets['apiseqs'][i]] * data_len, self.conf.apiseq_len)
            tokens = self.pad([self._eval_sets['tokens'][i]] * data_len, self.conf.tokens_len)
            n_good = K

            sims = model.predict([methnames, apiseqs, tokens, descs], batch_size=data_len).flatten()
            r = rankdata(sims, method='max')
            max_r = np.argmax(r)
            max_n = np.argmax(r[:n_good])
            c_1 += 1 if max_r == max_n else 0
            c_2 += 1 / float(r[max_r] - r[max_n] + 1)

        top1 = c_1 / float(data_len)
        # percentage of predicted most similar desc that is really the corresponding desc
        mrr = c_2 / float(data_len)

        return top1, mrr

    def eval(self, model, poolsize, K):
        """
        validate in a code pool.
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """

        def ACC(real, predict):
            sum = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1: sum = sum + 1
            return sum / float(len(real))

        def MAP(real, predict):
            sum = 0.0
            for id, val in enumerate(real):
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1: sum = sum + (id + 1) / float(index + 1)
            return sum / float(len(real))

        def MRR(real, predict):
            sum = 0.0
            for val in real:
                try:
                    index = predict.index(val)
                except ValueError:
                    index = -1
                if index != -1: sum = sum + 1.0 / float(index + 1)
            return sum / float(len(real))

        def NDCG(real, predict):
            dcg = 0.0
            idcg = IDCG(len(real))
            for i, predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance = 1
                    rank = i + 1
                    dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
            return dcg / float(idcg)

        def IDCG(n):
            idcg = 0
            itemRelevance = 1
            for i in range(n):
                idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
            return idcg

        # load valid dataset
        if self._eval_sets is None:
            methnames, apiseqs, tokens, descs = self.load_valid_data(poolsize)
            self._eval_sets = dict()
            self._eval_sets['methnames'] = methnames
            self._eval_sets['apiseqs'] = apiseqs
            self._eval_sets['tokens'] = tokens
            self._eval_sets['descs'] = descs
        acc, mrr, map, ndcg = 0, 0, 0, 0
        data_len = len(self._eval_sets['descs'])
        for i in range(data_len):
            print(i)
            desc = self._eval_sets['descs'][i]  # good desc
            descs = self.pad([desc] * data_len, self.conf.desc_len)
            methnames = self.pad(self._eval_sets['methnames'], self.conf.methname_len)
            apiseqs = self.pad(self._eval_sets['apiseqs'], self.conf.apiseq_len)
            tokens = self.pad(self._eval_sets['tokens'], self.conf.tokens_len)
            n_results = K
            sims = model.predict([methnames, apiseqs, tokens, descs], batch_size=data_len).flatten()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)  # predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            acc += ACC(real, predict)
            mrr += MRR(real, predict)
            map += MAP(real, predict)
            ndcg += NDCG(real, predict)
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map = map / float(data_len)
        ndcg = ndcg / float(data_len)

        return acc, mrr, map, ndcg

    def repr_code(self, model):
        methnames, apiseqs, tokens = self.load_use_data()
        methnames = self.pad(methnames, self.conf.methname_len)
        apiseqs = self.pad(apiseqs, self.conf.apiseq_len)
        tokens = self.pad(tokens, self.conf.tokens_len)

        vecs = model.repr_code([methnames, apiseqs, tokens], batch_size=1000)
        vecs = vecs.astype('float32')
        vecs = normalize(vecs)
        self.save_code_reprs(vecs)
        return vecs

    def search(self, model, query, n_results=10):
        desc = [self.convert(self.vocab_desc, query)]  # convert desc sentence to word indices
        padded_desc = self.pad(desc, self.conf.desc_len)
        desc_repr = model.repr_desc([padded_desc])
        desc_repr = desc_repr.astype('float32')

        codes = []
        sims = []
        threads = []
        for i, code_reprs_chunk in enumerate(self.code_reprs):
            t = threading.Thread(target=self.search_thread,
                                 args=(codes, sims, desc_repr, code_reprs_chunk, i, n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:  # wait until all sub-threads finish
            t.join()
        return codes, sims

    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):
        # 1. compute similarity
        chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)

        # 2. choose top results
        negsims = np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]
        chunk_codes = [self.code_base[i][k] for k in maxinds]
        chunk_sims = chunk_sims[0][maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)

    def postproc(self, codes_sims):
        codes_, sims_ = zip(*codes_sims)
        codes = [code for code in codes_]
        sims = [sim for sim in sims_]
        final_codes = []
        final_sims = []
        n = len(codes_sims)
        for i in range(n):
            is_dup = False
            for j in range(i):
                if codes[i][:80] == codes[j][:80] and abs(sims[i] - sims[j]) < 0.01:
                    is_dup = True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(sims[i])
        return zip(final_codes, final_sims)



if __name__ == '__main__':
    conf = configs.conf()
    codesearcher = CodeSearcher(conf)

    mode = 'train'

    #  Define model
    model = eval(conf.model_name)(conf)
    model.build()
    optimizer = conf.optimizer
    model.compile(optimizer=optimizer)

    if mode == 'train':
        codesearcher.train(model)

    elif mode == 'eval':
        # evaluate for a particular epoch
        # load model
        if conf.reload > 0:
            codesearcher.load_model_epoch(model, conf.reload)
        codesearcher.eval(model, -1, 10)

    elif mode == 'repr_code':
        # load model
        if conf.reload > 0:
            codesearcher.load_model_epoch(model, conf.reload)
        vecs = codesearcher.repr_code(model)

    elif mode == 'search':
        # search code based on a desc
        if conf.reload > 0:
            codesearcher.load_model_epoch(model, conf.reload)
        codesearcher.load_code_reprs()
        codesearcher.load_codebase()
        while True:
            try:
                query = input('Input Query: ')
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input:")
                traceback.print_exc()
                break
            codes, sims = codesearcher.search(model, query, n_results)
            zipped = zip(codes, sims)
            zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
            zipped = codesearcher.postproc(zipped)
            zipped = list(zipped)[:n_results]
            results = '\n\n'.join(map(str, zipped))  # combine the result into a returning string
            print(results)
