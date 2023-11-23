from models.directed_graph_model.adpa import ADPA

class ModelZoo():
    def __init__(self, logger, args, num_nodes, feat_dim, output_dim, task_level=None):
        super(ModelZoo, self).__init__()
        self.logger = logger
        self.args = args
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.prop_steps = args.prop_steps
        self.q = self.args.edge_q if self.task_level == "edge" else self.args.node_q
        self.log_model()
    def log_model(self):
        
        if self.args.model_name == "adpa":
            self.logger.info(f"model: {self.args.model_name}, r: {self.args.r}, hidden_dim: {self.args.hidden_dim}, dropout: {self.args.dropout}")

    def model_init(self):
        if self.args.model_name == "adpa":
            model = ADPA(r=self.args.r,
                        feat_dim=self.feat_dim, hidden_dim=self.args.hidden_dim, output_dim=self.output_dim, 
                        dropout=self.args.dropout, num_nodes=self.num_nodes, prop_steps=self.prop_steps, num_layers=self.args.num_layers,
                        task_level=self.task_level)
        else:
            return NotImplementedError
        return model