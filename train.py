import dataset
from torchmeta.utils.data import BatchMetaDataLoader

train_data = dataset.FewRel(num_classes_per_task=5)
train_data_loader = BatchMetaDataLoader(train_data, batch_size=16, num_workers=4)

for batch in train_data_loader:
    print(batch)
    import pdb; pdb.set_trace()
