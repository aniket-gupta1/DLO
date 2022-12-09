import os

class Config(object):
    def __init__(self, d):
        super(Config, self).__init__()
        self.input_dim = d
        self.dim_Q = d
        self.dim_K = d
        self.dim_V = d
        self.num_heads = 4
        self.ff_dim = 10
        self.num_cells = 1
        self.dropout = 0.2

        # Output
        self.output_dim = 6

        # Dataloader
        if os.getlogin() == 'ngc':
            self.root = "/home/ngc/SemSeg/Datasets/KITTI_short/dataset"
        elif os.getlogin() == 'aniket':
            self.root = "/media/aniket/77b12655-2aa3-4049-ac40-36a0a306712a/SemanticKITTI/dataset"
        else:
            self.root = ""
        self.validation_seq = 8
        self.inbetween_poses = True
        self.form_transformation = True

        # Optimizer
        self.lr = 0.0001
        self.weight_decay = 0.0001

        # training
        self.num_epochs = 10000

        # model
        self.model_type = "cross_attention"

        self.use_random_sampling = False
        self.regress_transformation = False
        self.embed_size = d

        self.downsampled_features = 500
