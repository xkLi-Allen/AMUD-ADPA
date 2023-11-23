def add_model_config(parser):
    parser.add_argument('--model_name', help='gnn model', type=str, default="adpa")
    parser.add_argument('--num_layers', help='number of gnn layers', type=int, default=1)
    parser.add_argument('--dropout', help='drop out of gnn model', type=float, default=0.64251)
    parser.add_argument('--hidden_dim', help='hidden units', type=int, default=32)
    parser.add_argument('--edge_dim', help='hidden units of linearR model in edge-level tasks', type=int, default=2)
    # scalable gnn model
    parser.add_argument('--prop_steps', help='prop steps', type=int, default=2)
    # adj normalize
    parser.add_argument('--r', help='symmetric normalized unit', type=float, default=0.42749)
    # not included in the current experiment
    parser.add_argument('--node_q', help='the imaginary part of the complex unit in node-level tasks', type=float, default=0.25)
    parser.add_argument('--edge_q', help='the imaginary part of the complex unit in edge-level tasks', type=float, default=0.25)
    # not included in the current experiment
    parser.add_argument('--ppr_alpha', help='ppr approxmite symmetric adj unit', type=float, default=0.1)
    parser.add_argument('--message_alpha', help='weighted message operator', type=float, default=0.5)
    # not included in the current experiment
    parser.add_argument('--neighbor_hops', help='number of hops to consider', type=int, default=3)
    # not included in the current experiment
    parser.add_argument('--filter_order', help='order of filter process', type=int, default=10)
    # not included in the current experiment
    parser.add_argument('--nste_alpha', help='propagation coefficient of nste', type=float, default=0.4)
    parser.add_argument('--nste_beta', help='propagation coefficient of nste', type=float, default=0.2)
    # not included in the current experiment
    parser.add_argument('--poly_order', help='the order of polynomial', type=int, default=3)
    # not included in the current experiment
    parser.add_argument('--init_poly_coeff', help=' the parameter to initialize polynomial coefficients', type=float, default=1)
    # not included in the current experiment
    parser.add_argument('--gpr_alpha', help='propagation class for ppr-link GPR_GNN', type=float, default=0.1)
