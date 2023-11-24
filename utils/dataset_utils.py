import os

def download_dataset(dataset_name, dataset_type, dataset_size):
    filename = None
    if dataset_name == "aishell-4":
        if dataset_type == "train":
            filename = f"train_{dataset_size}.tar.gz"
            os.system(f"wget https://us.openslr.org/resources/111/train_{dataset_size}.tar.gz")
        elif dataset_type == "test":
            filename = "test.tar.gz"
            os.system("wget https://us.openslr.org/resources/111/test.tar.gz")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    
    os.makedirs("data", exist_ok=True)
    os.system(f"tar -xvf {filename} -C data/{dataset_name}")
    os.system(f"rm {filename}")

    # return the path tree of the dataset
    return os.path.join("data", dataset_name, dataset_type)