import json
import os
import numpy as np
from torchmeta.utils.data import ClassDataset, CombinationMetaDataset, Dataset
from torchmeta.transforms import ClassSplitter
import encoder

enc = encoder.RobertaSentenceEncoder(max_length=128)

class FewRel(CombinationMetaDataset):
    def __init__(self, root="data", num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split="train",
                 use_vinyals_split=True, target_transform=None,):
        class_splitter = ClassSplitter(num_train_per_class=5, random_state_seed=10)
        dataset = FewRelClassDataset(root, meta_split=meta_split, use_vinyals_split=use_vinyals_split)
        super().__init__(dataset, num_classes_per_task, target_transform=target_transform, dataset_transform=class_splitter)


class FewRelClassDataset(ClassDataset):

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split=None,
                 use_vinyals_split=True, transform=None, class_augmentations=None):
        super().__init__(meta_split=meta_split)
        self.root = root
        filename = "train_wiki.json"
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{filename} does not exist at {root}!")
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        np.random.seed = 42

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, index):
        class_name = self.classes[index % self.num_classes]
        data = self.data[0][class_name]
        transform = self.encoder.tokenize
        class_id = self.read_labels().index(class_name)
        return FewRelDataset(index, data, class_id, transform=transform)

    def __len__(self):
        return 1000000000


class FewRelDataset(Dataset):

    def __init__(self, index, data: str, class_id: str):
        super().__init__(index, data)
        self.data = data
        self.class_id = class_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = ' '.join(self.data[index]["tokens"])
        head = self.data[index]['h']
        tail = self.data[index]['t']
        return sentence, head, tail, self.class_id

