import tensorflow as tf
class clstm_config(object):
    """Wrapper calss for C-LSTM hyperparameter"""

    def __init__(self):
        #data hyperparameter
        self.data_window = 1 #dimention of data window
        self.feature_size = 1 #dimention of feature of each data
        self.padding = 1
        self.is_shuffle = True
        self.data_file_path = "aiopsdata" #aiopsdata ,A1Benchmark, A2Benchmark, A3Benckmark, A4Benchmark
        self.is_oversampling = True
        self.oversampling_method = 'VAE' #SMOTE , ADASYN or VAE
        self.train_percentage = 0.7

        self.kpi_id = 'da403e4e3f87c9e0' #02e99bd4f6cfb33f,1c35dbf57f55f5e4,18fbb1d5a5dc099d,da403e4e3f87c9e0
        self.kpi_file = 'phase1'

        # autoencoder
        self.enable_denoised_autoencoder = False

        #model hyperparameter
        self.model = 'clstm'
        self.num_classes = 2 #dimention of class
        self.filter_size = 4 #CNN filter window size
        self.num_filters = 64 #CNN filter number
        self.hidden_size = 64 #dimension of hidden unit in LSTM
        self.num_layers = 2 #number for layers in LSTM, 2 means biLSTM
        self.dense_size = 32 #dimension of fully connected layer
        self.l2_reg_lambda = 0.0001 #L2 regularization strength
        self.keep_prob = 0.5 #Dropout keep probability'

        #training hyperparameter
        self.learning_rate = 0.0002 #learning rate 1e-1,1e-2,0.2,0.5
        self.decay_steps = 100000 #Learning rate decay steps
        self.decay_rate = 1 #Learning rate decay rate. Range: (0, 1]
        self.batch_size = 1 #Batch size 2*x 128 ,fix batch_size firstly
        self.num_epochs = 1 #Number of epochs 20,50,80
        self.save_every_steps = 1000 #Save the model after this many steps
        self.num_checkpoint = 10 #Number of models to store'
        self.evaluate_every_steps = 100 #Evaluate the model on validation set after this many steps

