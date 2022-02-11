class Config(object):
    def __init__(self):
        # directories
        self.save_dir = ''
        self.log_dir = ''
        self.train_data_file = ''
        self.val_data_file = ''

        # define variables
        self.n_classes = 2
        self.train_ratio = 0.8
        self.batch_size = 128
