import os
import os.path as osp

from datasets.utils import file_exist

# Base class for node-level tasks
class NodeDataset:
    def __init__(self, root, name, k):
        self.k = k
        self.name = name.lower() if name != "papers100m" else "papers100M"
        self.root = osp.join(root, self.name)
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        self.data = None
        self.train_idx, self.val_idx, self.test_idx, self.seed_idx, self.stopping_idx = None, None, None, None, None
        self.preprocess()

    @property
    def raw_file_paths(self):
        raise NotImplementedError

    @property
    def processed_file_paths(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def preprocess(self):
        print(self.raw_file_paths)
        if file_exist(self.raw_file_paths):
            # logger.info("Files already downloaded.")
            pass
        else:
            print("Downloading...")
            if not file_exist(self.raw_dir):
                os.makedirs(self.raw_dir)
            self.download()
            print("Downloading done!")

        if file_exist(self.processed_file_paths):
            # logger.info("Files already processed.")
            pass
        else:
            print("Processing...")
            if not file_exist(self.processed_dir):
                os.makedirs(self.processed_dir)
            self.process()
            print("Processing done!")
