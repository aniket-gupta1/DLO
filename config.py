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
        self.output_dim = 7

        # Dataloader
        # self.root = "/home/ngc/SemSeg/Datasets/SemanticKITTI/dataset"
        self.root = "/media/aniket/77b12655-2aa3-4049-ac40-36a0a306712a/SemanticKITTI/dataset"
        self.validation_seq = 8
        self.inbetween_poses = True
        self.form_transformation = True

        # Optimizer
        self.lr = 0.0001
        self.weight_decay = 0.0001

        # training
        self.num_epochs = 10

        # model
        self.model_type = "cross_attention"