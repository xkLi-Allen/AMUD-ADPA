import json
import torch
import numpy as np
import os.path as osp
import pickle as pkl
import scipy.sparse as sp

from itertools import chain
from torch_sparse import coalesce
from datasets.base_data import Graph
from datasets.base_dataset import NodeDataset
from datasets.link_split import link_class_split
from datasets.node_split import node_class_split
from datasets.utils import pkl_read_file, download_to, remove_self_loops, coomatrix_to_torch_tensor, edge_homophily, node_homophily, linkx_homophily, set_spectral_adjacency_reg_features, directional_feature_smoothness, directional_label_smoothness, even_quantile_labels, adjusted_homophily, li_node
import scipy
import scipy.io
from scipy.io import loadmat
from torch import Tensor

class UsDUwPyGSDDataset(NodeDataset):
    '''
    Dataset description: arXiv'22 PyTorch Geometric Signed Directed: A Software Package on Graph Neural Networks for Signed and Directed Graphs, PyGSD https://arxiv.org/pdf/2202.10793.pdf.
    Notably, in the version of the directed networks, we select the original Chameleon, Squirrel datasets from ICLR'20 GEOM-GCN: GEOMETRIC GRAPH CONVOLUTIONAL NETWORKS, https://arxiv.org/pdf/2002.05287.pdf and NeurIPS'22 Revisiting Heterophily For Graph Neural Networks, https://arxiv.org/pdf/2210.07606.pdf.

    -> CoraML:      unsigned & directed & unweighted network.
    -> CiteSeer:    unsigned & directed & unweighted network. (Directed version of Planetoid (CiteSeer))
    -> Chameleon:   unsigned & directed & unweighted network. (Directed version of Chameleon)
    -> Squirrel:    unsigned & directed & unweighted network. (Directed version of Squirrel)
    -> WikiCS:      unsigned & directed & unweighted network.
    -> Slashdot:    unsigned & directed & unweighted network. (Unsigned version of Slashdot)
    -> Epinions:    unsigned & directed & unweighted network. (Unsigned version of Epinions)
    -> WikiTalk:    unsigned & directed & unweighted network.

    We remove the additional multiple and self-loop edges and normalize the graph to a directed network, hence the differences with the original report -> arXiv'22 PyTorch Geometric Signed Directed: A Software Package on Graph Neural Networks for Signed and Directed Graphs, PyGSD https://arxiv.org/pdf/2202.10793.pdf and NeurIPS'22 Revisiting Heterophily For Graph Neural Networks, https://arxiv.org/pdf/2210.07606.pdf.
    Edge
    -> CiteSeer:    4715 -> 4591
    -> Chameleon:   36,101 -> 36,051
    -> Squirrel:    217,073 -> 216,933
    -> WikiCS:      297,110 -> 290,519

    We remove the additional multiple and self-loop edges and normalize the graph to a directed network, hence the differences with the original report -> arXiv'22 PyTorch Geometric Signed Directed: A Software Package on Graph Neural Networks for Signed and Directed Graphs, PyGSD https://arxiv.org/pdf/2202.10793.pdf an AAAI'21 SDGNN: Learning Node Representation for Signed Directed Networks, https://arxiv.org/pdf/2101.02390.pdf.
    Node
    -> Slashdot:    82,140 -> 75,144
    -> Epinions:    131,580 -> 114,467

    CoraML:     popular citation networks with node labels corresponding to scientific subareas.
                2,995 nodes, 8,416 edges, 2879 feature dimensions, 7 classes num.
                Edge homophily: 0.7921, Node homophily:0.8079, Linkx homophily:0.7438.

    CiteSeer:   popular citation networks with node labels corresponding to scientific subareas.
                3,312 nodes, 4,591 edges, 3703 feature dimensions, 6 classes num. 
                Edge homophily: 0.7393, Node homophily:0.7251, Linkx homophily:0.6268.

    Chameleon:  Nodes represent articles from the English Wikipedia, and edges reflect mutual links between them. 
                Node features indicate the presence of particular nouns in the articles. 
                Nodes are grouped into five categories based on the average monthly traffic of the web page.
                2,277 nodes, 36,051 edges, 2325 feature dimensions, 5 classes num.
                Edge homophily: 0.2339, Node homophily:0.247, Linkx homophily:0.062.

    Squirrel:   Nodes represent articles from the English Wikipedia, and edges reflect mutual links between them. 
                Node features indicate the presence of particular nouns in the articles. 
                Nodes are grouped into five categories based on the average monthly traffic of the web page.
                5,201 nodes, 216,933 edges, 2089 feature dimensions, 5 classes num.
                Edge homophily: 0.2234, Node homophily:0.2156, Linkx homophily:0.0254.

    WikiCS:     Collection of Computer Science articles.
                11,701 nodes, 290,519 edges, 300 feature dimensions, 10 classes num.
                Edge homophily: 0.6885, Node homophily:0.6743, Linkx homophily:0.597.

    Slashdot:   Slashdot is from a technology-related news website with user communities. 
                The website introduced Slashdot Zoo features that allow users to tag each other as friends or foes. 
                The dataset is a common signed social network with friends and enemies labels.
                In our experiments, we consider only friendships.
                75,144 nodes, 425,072 edges, 100 feature dimensions, No labels

    Epinions:   It is a who-trust-whom online social network of a consumer review site Epinions.com. 
                Members of the site can indicate their trust or distrust of the reviews of others. 
                The network reflects people's opinions on others.
                In our experiments, we consider only trust relationship.
                114,467 nodes, 717,129 edges, 100 feature dimensions, No labels

    WikiTalk:   It contains all users and discussion from the inception of Wikipedia until Jan. 2008. 
                Nodes in the network represent Wikipedia users 
                Directed edge from node vi to node vj denotes that user i edited at least once a talk page of user j. 
                2,388,953 nodes, 5,018,445 edges, 100 feature dimensions, No labels

    
    Node split:      
        ogbn-arxiv:
            official:   We propose to train on papers published until 2017, 
                        validate on those published in 2018, 
                        and test on those published since 2019.
                        train/val/test = 90,941/29,799/48,603
        CoraML & CiteSeer
            official:
                train_idx: classes num * 20
                val_idx: 500
                test_idx: all remain nodes: 2995 - (7 * 20) - 500 = 2355
                seed_idx: 0
        Chameleon & Squirrel
            official:
                train_idx/val/test = 60%/20%/20%
                (10 random data split results)
        WikiCS
        We split the nodes in each class into two sets, 50% for the test set and 50% potentially visible. 
        From the visible set, we generated 20 different splits of training, validation and early-stopping sets: 
        5% of the nodes in each class were used for training in each split, 
        22.5% were used to evaluate the early-stopping criterion, and 
        22.5% were used as the validation set for hyperparameter tuning. 
            official:
                train/val/test/stopping = 580/1769/5847/3505
                (20 random data split results)

    Edge split:
        CoraML & CiteSeer & Chameleon & Squirrel & WikiCS & Slashdot & Epinions & arxiv & WikiTalk
            official:
                train_idx/val/test = 80%/15%/5%
                (10 random data split results)
    '''
    def __init__(self, args, name="coraml", root="./datasets/directed/unweighted/", k=2,
                 node_split="official", node_split_id=0, edge_split="direction", edge_split_id=0):
        super(UsDUwPyGSDDataset, self).__init__(root + 'pygsd/', name, k)
        self.read_file()
        self.split_id = 0
        self.node_split = node_split
        self.node_split_id = node_split_id
        self.edge_split = edge_split
        self.edge_split_id = edge_split_id
        self.cache_node_split = osp.join(self.raw_dir, "{}-node-splits".format(self.name))
        self.cache_edge_split = osp.join(self.raw_dir, "{}-edge-splits".format(self.name))

        if self.name == "wikics":
            self.official_split = self.raw_file_paths[0]
        
        elif self.name in ("chameleondir", "squirreldir"):
            self.official_split = self.raw_file_paths[2:]

        else:
            self.official_split = None
        
        if self.name in ("proteins", "coauthor-cs", "coauthor-physic", "pubmed", "cornell", "texas", "wisconsin", "squirrel",
                "chameleon", "coramlundir", "citeseer", "minesweeper", "tolokers", "roman", "rating", "question", "squirrelfilter",
                "chameleonfilter", "actor", "cornellundir", "romanundir", "ratingundir", "snap"):
            self.edge_split = "existence"

        if self.name in ("roman", "rating", "minesweeper", "tolokers", "question", "squirrelfilterdir", "chameleonfilterdir", "snap", "actor", "texas", "cornell", "wisconsin"):
            self.train_idx, self.val_idx, self.test_idx = self.generate_split()

        elif self.name not in ("wikitalk", "slashdot", "epinions"):
            self.train_idx, self.val_idx, self.test_idx, self.seed_idx, self.stopping_idx = node_class_split(name=name.lower(), data=self.data, 
                                                                                        cache_node_split=self.cache_node_split,
                                                                                        official_split=self.official_split,
                                                                                        split=self.node_split, node_split_id=self.node_split_id, 
                                                                                        train_size_per_class=20, val_size=500)
        
        edge_index = torch.from_numpy(np.vstack((self.edge.row.numpy(), self.edge.col.numpy()))).long()
        self.observed_edge_idx, self.observed_edge_weight, self.train_edge_pairs_idx, self.val_edge_pairs_idx, self.test_edge_pairs_idx, self.train_edge_pairs_label, self.val_edge_pairs_label, self.test_edge_pairs_label\
        = link_class_split(edge_index=edge_index, A=self.edge.sparse_matrix,
                        cache_edge_split=self.cache_edge_split, 
                        task=self.edge_split, edge_split_id=self.edge_split_id,
                        prob_val=0.15, prob_test=0.05, )
        self.num_node_classes = self.num_classes
        if edge_split in ("existence", "direction", "sign"):
            self.num_edge_classes = 2
        elif edge_split in ("three_class_digraph"):
            self.num_edge_classes = 3
        elif edge_split in ("four_class_signed_digraph"):
            self.num_edge_classes = 4
        elif edge_split in ("five_class_signed_digraph"):
            self.num_edge_classes = 5
        else:
            self.num_edge_classes = None
        if args.heterogeneity and self.name not in ("wikitalk", "slashdot", "epinions"):
            self.edge_homophily = edge_homophily(self.adj, self.y)
            self.node_homophily = node_homophily(self.adj, self.y)
            self.linkx_homophily = linkx_homophily(self.adj, self.y)
            self.adjusted_homophily = adjusted_homophily(self.adj, self.y)
            self.li_homophily = li_node(self.adj, self.y)
        if self.name not in ("wikitalk", "slashdot", "epinions"):
            self.edge_sim, self.no_edge_sim, self.sim = directional_feature_smoothness(self.adj, self.x)

    @property
    def raw_file_paths(self):
        dataset_name = {
            'coraml': 'cora_ml.npz',
            'citeseerdir': 'citeseer.npz',
            'wikitalk': 'wikitalk.npz',
            'slashdot': 'slashdot.csv',
            'epinions': 'epinions.csv',
            'amazon-computers': 'amazon_electronics_computers.npz',
            'amazon-photo': 'amazon_electronics_photo.npz',
            'coauthor-cs': 'ms_academic_cs.npz', 
            'coauthor-physic': 'ms_academic_phy.npz',
            'pubmed': 'pubmed.npz',
            'minesweeper':'minesweeper.npz',
            'tolokers':'tolokers.npz',
            'roman':'roman_empire.npz',
            'rating':'amazon_ratings.npz',
            'question':'questions.npz',
            'squirrelfilterdir':'squirrel_filtered_directed.npz',
            'chameleonfilterdir':'chameleon_filtered_directed.npz',
            'squirrelfilter':'squirrel_filtered.npz',
            'chameleonfilter':'chameleon_filtered.npz',
            'snap': 'snap_patents.mat',
            'actor': 'actor.npz',
            'texas': 'texas.npz',
            'cornell': 'cornell.npz',
            'wisconsin': 'wisconsin.npz'
        }
        splits_name = {
            'snap': 'snap-patents-splits.npy',
            'pokec': 'pokec-splits.npy',
        }
        if self.name in ("coraml", "citeseerdir", "wikitalk", "slashdot", "epinions"):
            filename = dataset_name[self.name]
            return [osp.join(self.raw_dir, filename)]
        elif self.name in ("squirrel", "chameleon", "coramlundir", "citeseer", "squirrelfilter", "chameleonfilter", "cornellundir", "romanundir", "ratingundir"):
            filenames = ["out1_node_feature_label.txt", "out1_graph_edges.txt"]
            return [osp.join(self.raw_dir, filename) for filename in filenames]
        elif self.name in ("chameleondir", "squirreldir"):
            filenames = ["out1_node_feature_label.txt", "out1_graph_edges.txt"]
            for i in range(10):
                filenames.append("{}_split_0.6_0.2_{}.npz".format(self.name[:-3], i))
            return [osp.join(self.raw_dir, filename) for filename in filenames]
        
        elif self.name == "wikics":
            filenames = ["data.json", "metadata.json", "statistics.json"]
            return [osp.join(self.raw_dir, filename) for filename in filenames]
        
        elif self.name == "proteins":
            filenames = ["PROTEINS_A.txt", "PROTEINS_node_attributes.txt", "PROTEINS_node_labels.txt"]
            return [osp.join(self.raw_dir, filename) for filename in filenames]
            
        elif self.name in ("amazon-computers", "amazon-photo", "pubmed", "coauthor-cs", "coauthor-physic", "minesweeper", "tolokers", "roman", "rating", "question", "squirrelfilterdir", "chameleonfilterdir", "actor", "texas", "cornell", "wisconsin"):
            filename = dataset_name[self.name]
            return [osp.join(self.raw_dir, filename)]
        elif self.name in ("snap"):
            filenames = [dataset_name[self.name]]
            if self.name in splits_name:
                filenames += [splits_name[self.name]]
            return [osp.join(self.raw_dir, filename) for filename in filenames]
        
    @property
    def processed_file_paths(self):
        return osp.join(self.processed_dir, f"{self.name}.graph")

    def read_file(self):
        self.data = pkl_read_file(self.processed_file_paths)
        self.edge = self.data.edge
        self.node = self.data.node
        self.x = self.data.x
        self.y = self.data.y
        self.adj = self.data.adj
        self.edge_type = self.data.edge_type
        self.num_features = self.data.num_features 
        self.num_classes = self.data.num_classes
        self.num_node = self.data.num_node
        self.num_edge = self.data.num_edge

    def download(self):

        dataset_drive_url = {
            'coraml': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/cora_ml.npz',
            'citeseerdir': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/citeseer.npz',
            'wikitalk': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/wikitalk.npz',
            'wikics': 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset',
            'chameleondir': 'https://github.com/SitaoLuan/ACM-GNN/tree/main/new_data/chameleon',
            'slashdot': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/slashdot.csv',
            'epinions': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/datasets/epinions.csv',
            'squirreldir': 'https://github.com/SitaoLuan/ACM-GNN/tree/main/new_data/squirrel',
        }
        file_url = dataset_drive_url[self.name]
        if self.name in ("coraml", "citeseerdir", "wikitalk", "slashdot", "epinions"):
            print("Download:{} to {}".format(file_url, self.raw_file_paths[0]))
            download_to(file_url, self.raw_file_paths[0])

        elif self.name in ("chameleondir", "squirreldir"):
            print("Download:{} to {}".format(file_url + "/out1_node_feature_label.txt", self.raw_file_paths[0]))
            download_to(file_url + "/out1_node_feature_label.txt", self.raw_file_paths[0])
            print("Download:{} to {}".format(file_url + "/out1_graph_edges.txt", self.raw_file_paths[1]))
            download_to(file_url + "/out1_graph_edges.txt", self.raw_file_paths[1])
            split_drive_url = 'https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/splits'
            for i in range(10):
                print("Download:{} to {}".format(split_drive_url + "/{}_split_0.6_0.2_{}.npz".format(self.name[:-3], i), self.raw_file_paths[i+2]))
                download_to(split_drive_url + "/{}_split_0.6_0.2_{}.npz".format(self.name[:-3], i), self.raw_file_paths[i+2])               

        elif self.name == "wikics":
            print(file_url + "/data.json", self.raw_file_paths[0])
            print("Download:{} to {}".format(file_url + "/metadata.json", self.raw_file_paths[1]))
            download_to(file_url + "/metadata.json", self.raw_file_paths[1])
            print("Download:{} to {}".format(file_url + "/statistics.json", self.raw_file_paths[2]))
            download_to(file_url + "/statistics.json", self.raw_file_paths[2])
            
                                
    def process(self):
        if self.name in ("coraml", "citeseerdir"):
            with np.load(self.raw_file_paths[0], allow_pickle=True) as loader:
                loader = dict(loader)
                edge_index = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                        loader['attr_indptr']), shape=loader['attr_shape'])
                labels = loader.get('labels')

            edge_index = edge_index.tocoo()
            edge_index = coomatrix_to_torch_tensor(edge_index)
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index

            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"

            features = torch.from_numpy(features.todense()).float()
            num_node = features.shape[0]
            labels = torch.from_numpy(labels).long()
            
        elif self.name in ("amazon-computers", "amazon-photo", "pubmed", "coauthor-cs", "coauthor-physic"):
            with np.load(self.raw_file_paths[0], allow_pickle=True) as loader:
                loader = dict(loader)
                edge_index = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape = loader['adj_shape'])
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape = loader['attr_shape'])
                labels = loader.get('labels')
                
            edge_index = edge_index.tocoo()
            edge_index = coomatrix_to_torch_tensor(edge_index)
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index
            
            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
            
            features = torch.from_numpy(features.todense()).float()
            num_node = features.shape[0]
            labels = torch.from_numpy(labels).long()
            
        elif self.name in ("minesweeper", "tolokers", "roman", "rating", "question", "squirrelfilterdir", "chameleonfilterdir", "actor", "texas", "cornell", "wisconsin"):
            with np.load(self.raw_file_paths[0], allow_pickle=True) as loader:
                loader = dict(loader)
                features = torch.tensor(loader['node_features'], dtype=torch.float)
                edge_index = torch.tensor(loader['edges'],dtype=torch.int64).t().contiguous()
                labels = torch.tensor(loader.get('node_labels'), dtype=torch.int64)
            
            num_node = features.shape[0]
            edge_index, _ = coalesce(edge_index, None, features.size(0), features.size(0))
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            
            row, col = undi_edge_index
            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"

        elif self.name in ("snap"):
            nclass = 5
            fulldata = scipy.io.loadmat(self.raw_file_paths[0])
            edge_index = fulldata['edge_index']
            if not isinstance(edge_index, Tensor):
                edge_index = torch.from_numpy(edge_index)
            edge_index = torch.unique(edge_index, dim=1)
            features = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
            num_node = int(fulldata['num_nodes'])
            years = fulldata['years'].flatten()
            label = even_quantile_labels(years, nclass, verbose=False)
            labels = torch.tensor(label, dtype=torch.long)
            if self.name in ("snap","pokec"):
                edge_index = torch.unique(edge_index, dim=1)
            edge_index = remove_self_loops(edge_index)[0]
            row, col = edge_index
            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
            
        elif self.name in ("chameleondir", "squirreldir", "squirrel", "chameleon",
                           "coramlundir", "citeseer", "squirrelfilter", "chameleonfilter", "actor", "cornellundir", "romanundir", "ratingundir"):
            with open(self.raw_file_paths[0], 'r') as f:
                ori_data = f.read().split('\n')[1:-1]
                features = [[float(v) for v in r.split('\t')[1].split(',')] for r in ori_data]
                features = torch.tensor(features, dtype=torch.float)
                num_node = features.shape[0]

                labels = [int(r.split('\t')[2]) for r in ori_data]
                labels = torch.tensor(labels, dtype=torch.int64)

            with open(self.raw_file_paths[1], 'r') as f:
                ori_data = f.read().split('\n')[1:-1]
                ori_data = [[int(v) for v in r.split('\t')] for r in ori_data]
                edge_index = torch.tensor(ori_data, dtype=torch.int64).t().contiguous()
                edge_index, _ = coalesce(edge_index, None, features.size(0), features.size(0))

            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index

            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
            
        elif self.name == "proteins":
            with open(self.raw_file_paths[1], 'r') as f:
                ori_data = f.read().split('\n')
                #print(ori_data)
                features = [[float(r)] for r in ori_data]
                features = torch.tensor(features, dtype = torch.int64)
                num_node = features.shape[0]
                
            with open(self.raw_file_paths[2], 'r') as f:
                ori_data = f.read().split('\n')
                labels = [int(r) for r in ori_data]
                labels = torch.tensor(labels, dtype = torch.int64)
        
            with open(self.raw_file_paths[0], 'r') as f:
                ori_data = f.read().split('\n')
                ori_data = [[int(v)-1 for v in r.split(", ")] for r in ori_data]
                edge_index = torch.tensor(ori_data, dtype=torch.int64).t().contiguous()
                edge_index, _ = coalesce(edge_index, None, features.shape[0], features.shape[0])
                
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index
            
            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
      
        elif self.name == "wikics":
            with open(self.raw_file_paths[0], 'r') as f:
                ori_data = json.load(f)

            features = torch.tensor(ori_data['features'], dtype=torch.float)
            labels = torch.tensor(ori_data['labels'], dtype=torch.long)
            num_node = features.shape[0]

            edges = [[(i, j) for j in js] for i, js in enumerate(ori_data['links'])]
            edges = list(chain(*edges))
            edges = np.array(edges).transpose()

            edge_index = torch.from_numpy(edges)
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            row, col = undi_edge_index

            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
     
        elif self.name == "wikitalk":
            adj = sp.load_npz(self.raw_file_paths[0])
            adj_coo = adj.tocoo()
            row, col = adj_coo.row, adj_coo.col
            edge_index = np.vstack((row, col))
            edge_index = torch.from_numpy(edge_index).long()
            undi_edge_index = torch.unique(edge_index, dim=1)
            undi_edge_index = remove_self_loops(undi_edge_index)[0]
            edge_index = undi_edge_index
            row, col = edge_index
            edge_weight = torch.ones(len(row))
            edge_type = "UDUw"
            edge_num_node = edge_index.max().item() + 1
            num_node = edge_num_node
            features = set_spectral_adjacency_reg_features(edge_num_node, edge_index, edge_weight, self.k)
            labels = None

        elif self.name in ("slashdot", "epinions"):
            data = []
            edge_weight = []
            edge_index = []
            node_map = {}
            with open(self.raw_file_paths[0], 'r') as f:
                for line in f:
                    x = line.strip().split(',')
                    if float(x[2]) >= 0:
                        assert len(x) == 3
                        a, b = x[0], x[1]
                        if a not in node_map:
                            node_map[a] = len(node_map)
                        if b not in node_map:
                            node_map[b] = len(node_map)
                        a, b = node_map[a], node_map[b]
                        data.append([a, b])

                        edge_weight.append(float(x[2]))

                edge_index = [[i[0], int(i[1])] for i in data]
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                undi_edge_index = torch.unique(edge_index, dim=1)
                undi_edge_index = remove_self_loops(undi_edge_index)[0]
                edge_index = undi_edge_index
                row, col = edge_index
                edge_weight = torch.ones(len(row))
                edge_type = "UDUw"
                edge_num_node = edge_index.max().item() + 1
                num_node = edge_num_node
                features = set_spectral_adjacency_reg_features(edge_num_node, edge_index, edge_weight, self.k)
                labels = None
        
        g = Graph(row, col, edge_weight, num_node, edge_type, x=features, y=labels)

        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def generate_split(self):
        have_split = ['roman', 'rating', 'minesweeper', 'tolokers', 'question', "squirrelfilterdir", "chameleonfilterdir", "squirrelfilter", "chameleonfilter", "snap", "actor", "texas", "cornell", "wisconsin"]
        if self.name in ("snap"):
            split_full = np.load(self.raw_file_paths[1], allow_pickle=True)
            split_idx = split_full[self.split_id]
            train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
            return train_idx, val_idx, test_idx

        elif self.name in have_split:
            ori_data = np.load(self.raw_file_paths[0])
            train_masks = torch.tensor(ori_data['train_masks'])
            val_masks = torch.tensor(ori_data['val_masks'])
            test_masks = torch.tensor(ori_data['test_masks'])

            train_mask = train_masks[self.split_id]
            val_mask = val_masks[self.split_id]
            test_mask = test_masks[self.split_id]

            train_idx_list = torch.where(train_mask)[0]
            val_idx_list = torch.where(val_mask)[0]
            test_idx_list = torch.where(test_mask)[0]

            return [train_idx for train_idx in train_idx_list], [val_idx for val_idx in val_idx_list], [test_idx for test_idx in test_idx_list]




