# small helper stuff
'''
    此代码定义了一个Bunch继承自内置dict类的类。该类Bunch允许将字典键作为属性进行访问。

    该类__init__的方法Bunch采用任意位置参数和关键字参数（分别为*args和**kwds），
    并使用该函数将它们传递给__init__该类的方法。这确保实例的行为类似于字典。
    dictsuperBunch

    调用该dict __init__方法后，实例__dict__的属性Bunch设置为自身。
    这会导致Bunch实例表现得像一个具有动态添加属性的对象。
'''
class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self
        
# SentenceVAE 
SVAE = Bunch(
    max_sequence_length = 300,
    latent_size = 100,
    rnn_types =  'GRU',
    bidirectional = True,
    num_layers = 1,
    hidden_size = 100,
    embedding_size = 100,
    word_dropout = 0.1,
    embedding_dropout = 0.1
    
)
train_cfg = Bunch(
    
    data_dir = '../data',    #The path to the directory where PTB data is stored, and auxiliary data files will be stored.
    create_data = True,        #If provided, new auxiliary data files will be created form the source data.
    max_sequence_length = 300, #Specifies the cut off of long sentences.
    min_occ = 1,             #If a word occurs less than "min_occ" times in the corpus, it will be replaced by the token.
    test  = False, #If provided   #performance will also be measured on the test set.
    embedding_size = 256,
    epochs=1000, #epochs
    batch_size=1024, #batch_size
    learning_rate=0.001, #learning_rate

    rnn_type = 'gru', #rnn_type Either 'rnn' or 'gru'.
    hidden_size = 256, #hidden_size
    num_layers = 1, #num_layers
    bidirectional = True, #bidirectional
    latent_size = 100, #latent_size
    word_dropout = 0, #word_dropout Word dropout applied to the input of the Decoder which means words will be replaced by <unk> with a probability of word_dropout.
    embedding_dropout = 0.3, #embedding_dropout Word embedding dropout applied to the input of the Decoder.

    anneal_function = 'logistic', #anneal_function Either 'logistic' or 'linear'.
    k = 0.0025, #k Steepness of the logistic annealing function.
    x0 = 2500, #x0 For 'logistic', this is the midpoint (i.e. when the weight is 0.5); for 'linear' this is the denominator.

    print_every = 10, #print_every
    tensorboard_logging = True, #tensorboard_logging If provided training progress is monitored with tensorboard.
    logdir = 'logs' ,#logdir Directory of log files for tensorboard.
    save_model_path = 'checkpoints'#save_model_path Directory where to store model checkpoints.
)