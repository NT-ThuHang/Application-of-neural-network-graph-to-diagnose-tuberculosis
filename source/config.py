from pathlib import Path
import torch
class Config(object):
    def __init__(self, args):
        # directories
        self.data = args.data
        self.save = args.save
        self.edge = args.edge
        self.setup()
        self.embedding = args.embedding
        self.checkpoint_path = self.save / 'model_checkpoint.ckpt' 
        self.logs_path = self.save / 'log.txt'
        self.message = args.message

        # define variables
        self.train_ratio = 0.8
        self.batch_size = 32

        #Hardware
        self.device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')

        #hyperparameter
        self.lr_step = args.lr_step
        self.lr = args.lr
        self.gamma = args.gamma
        self.input_channels = 2
        self.hidden_channels = [32, 32, 32]
        self.max_epoch = 200
        self.patience = 4

    def setup(self):
        if self.save is None:
            # generate to use if not provided
            root = Path('results')
            counter = 0
            while True:
                counter += 1
                path = root / 'result{:02d}'.format(counter)
                # if directory is not exist or empty
                if not path.exists() or not list(path.iterdir()):
                    path.mkdir(parents=True, exist_ok = True)
                    self.save = path
                    break
        print('Intermediate results will be stored in', str(self.save))

    def show(self):
        for attr, val in vars(self).items():
            print(attr+':', val)
