import zipfile
from torch.utils import data

def unzip_file(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def is_not_edgeless(data):
    return data.edge_index.shape[1] > 0      
        
def dataset_postprocessor(dataset, filter=None):
    if filter == "remove_edgeless_graphs":
        filter_fn = is_not_edgeless
    elif filter is None:
        filter_fn = lambda x: True
    else:
        raise Exception(f"Unknown filter {filter}")
    filtered_indexes = [i for i in range(len(dataset)) if filter_fn(dataset[i])]
    filtered_dataset = data.Subset(dataset, filtered_indexes)
    return filtered_dataset