import torch
from glob import glob

class Config(object):
    def __init__(self, args):
        # directories
        self.data = args.data
        self.save = args.save
        self.edge = args.edge
        self.embedding = args.embedding
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.data.is_dir():
            self.n_classes = len(glob(str(self.data)+'/*/'))
        else:
            #FIXME
            self.n_classes = 2
        self.setup()

        # define variables
        self.train_ratio = 0.8
        self.batch_size = 128

    def setup(self):
        if self.save is None:
            # generate to use if not provided
            if self.data.is_dir():
                root = self.data.parent
            else:
                root = self.data.parent.parent
            counter = 0
            while True:
                counter += 1
                path = root / 'result{:02d}'.format(counter)
                # if directory is not exist or empty
                if not path.exists() or not list(path.iterdir()):
                    path.mkdir(exist_ok = True)
                    self.save = path
                    break
        print('Intermediate results will be stored in', str(self.save))

    def show(self):
        for attr, val in vars(self).items():
            print(attr+':', val)