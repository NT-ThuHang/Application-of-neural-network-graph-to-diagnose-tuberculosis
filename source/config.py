import os
import torch
import glob

class Config(object):
    def __init__(self, args):
        # directories
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.edge = args.edge
        self.embedding = args.embedding
        self.preloader = args.preloader
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.check_paths()

        # define variables
        self.train_ratio = 0.8
        self.batch_size = 128

    @property
    def n_classes(self):
        # simply return the number of subdirectories
        return len(glob.glob(self.data_path+'/*/'))

    def check_paths(self):
        if self.preloader is None:
            if self.data_path is None:
                # Download if no data provided
                import zipfile 
                import gdown

                filename = gdown.download(id = '16k3T4iD5eOdM7tTHg1vq5st6mjLGbT8R')
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall()
                self.data_path = os.getcwd()+'/TB_Chest_Radiography_Database'
        else:
            # don't care about data if we use a preloader
            pass

        if self.save_path is None:
            self.save_path = os.getcwd()+'/Result'
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)


    def show(self):
        for attr, val in vars(self).items():
            print(attr+':', val)
        print('n_classes:', self.n_classes)
        




        

