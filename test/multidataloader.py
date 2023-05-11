from pprint import pprint
from utils.dataset_utils import MultiDataLoader

dataset_1 = [0,1,2,3,4,5,6,7,8,9]
dataset_2 = [10,11,12,13]
datasets = [dataset_1, dataset_2]
batch_sizes = [3, 2]
dl = MultiDataLoader(datasets, batch_sizes, 2)
n_epochs = 10
for epoch in range(n_epochs):
    for sample in dl:
        pprint(sample)
    print()