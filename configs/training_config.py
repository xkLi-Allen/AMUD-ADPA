def add_training_config(parser):
    # seed
    parser.add_argument('--seed', help='seed everything', type=int, default=2023)
    # use gpu
    parser.add_argument('--use_cuda', help='use gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', help='gpu id', type=int, default=1)
    parser.add_argument('--normalize_times', help='normalize train', type=int, default=1)
    # training details
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=500)
    parser.add_argument('--early_stop', help='early stop', type=int, default=50)
    parser.add_argument('--lr', help='learning rate of gnn model', type=float, default=0.06036)
    parser.add_argument('--weight_decay', help='weight decay of gnn model', type=float, default=0.01273)
    parser.add_argument('--train_batch_size', help='training batch size', type=int, default=1000)
    parser.add_argument('--eval_batch_size', help='val and test batch size', type=int, default=1000)
    parser.add_argument('--n_trials', help='the number of n_trials', type=int, default=1)
