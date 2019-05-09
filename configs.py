class conf:
    def __init__(self):
        self.data_dir = 'deepcs_dataset/'
        # train data
        self.train_methodname = 'train.methname.h5'
        self.train_apiseq = 'train.apiseq.h5'
        self.train_tokens = 'train.tokens.h5'
        self.train_desc = 'train.desc.h5'

        # valid data
        self.valid_methodname = 'test.methname.h5'
        self.valid_apiseq = 'test.apiseq.h5'
        self.valid_tokens = 'test.tokens.h5'
        self.valid_desc = 'test.desc.h5'

        # use data
        self.use_codebase = 'use.rawcode.txt'
        self.use_methodname = 'use.methname.h5'
        self.use_apiseq = 'use.apiseq.h5'
        self.use_tokens = 'use.tokens.h5'

        # use results data
        self.use_codevecs = 'use.codevecs.normalized.h5'

        # parameters
        self.methname_len = 6  # the max length of method name
        self.apiseq_len = 30
        self.tokens_len = 50
        self.desc_len = 30
        self.n_words = 10000  # the size of vocab

        # vocab_methname
        self.vocab_methname = 'vocab.methname.pkl'
        self.vocab_apiseq = 'vocab.apiseq.pkl'
        self.vocab_tokens = 'vocab.tokens.pkl'
        self.vocab_desc = 'vocab.desc.pkl'

        # model name
        self.model_name = 'JointEmbeddingModel'

        self.batch_size = 128
        self.chunk_size = 2000  ## load all data
        self.nb_epoch = 500
        self.validation_split = 0.2
        self.optimizer = 'adam'  ##学习率默认0.01
        self.valid_every = 5
        self.n_eval = 100
        self.save_every = 5
        self.reload = 0

        self.embed_dims = 100
        self.hidden_dims = 400
        self.lstm_dims = 200

        self.margin = 0.05
        self.sim_measure = 'cos'

        self.init_embed_weights_methodname = None
        self.init_embed_weights_tokens = None
        self.init_embed_weights_desc = None

