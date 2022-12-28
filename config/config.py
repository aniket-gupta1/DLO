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
            # self.root = "/home/ngc/SemSeg/Datasets/SemanticKITTI/dataset"
        elif os.getlogin() == 'aniket':
            self.root = "/media/aniket/77b12655-2aa3-4049-ac40-36a0a306712a/SemanticKITTI/dataset"
        else:
            self.root = ""
        self.validation_seq = 8
        self.inbetween_poses = True
        self.form_transformation = True
        self.downsample = False

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

        # Voxel subsampling
        self.voxel_size1 = 0.5
        self.voxel_size2 = 1.5
        self.use_voxel_sampling = False

        # Eval
        self.eval_time = 1

        # KPConv options
        self.num_layers = 2
        self.neighborhood_limits = [50, 50]
        self.aggregation_mode = "sum"
        self.first_subsampling_dl = 0.03  # Set smaller to have a higher resolution
        self.first_feats_dim = 512
        self.fixed_kernel_points = "center"
        self.in_feats_dim = 1
        self.in_points_dim = 3
        self.conv_radius = 2.75
        self.deform_radius = 5.0
        self.KP_extent = 2.0
        self.KP_influence = "linear"
        self.overlap_radius = 0.04
        self.use_batch_norm = True
        self.batch_norm_momentum = 0.02
        self.modulated = False
        self.num_kernel_points = 15
        self.architecture = ['simple',
                       'resnetb',
                       'resnetb',
                       'resnetb_strided',
                       'resnetb',
                       'resnetb', ]



