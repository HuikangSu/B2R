import os
import numpy as np
import collections
import pickle
import h5py
from tqdm import tqdm
import yaml


def hdf5_to_dict(hdf5_file_path):
    """
    Convert HDF5 file to Python dictionary.

    Args:
        hdf5_file_path (str): Path to the HDF5 file

    Returns:
        dict: Dictionary containing HDF5 file data
    """
    with h5py.File(hdf5_file_path, 'r') as file:
        data_dict = {}
        for key in file.keys():
            if isinstance(file[key], h5py.Dataset):
                data_dict[key] = file[key][:]
            elif isinstance(file[key], h5py.Group):
                data_dict[key] = hdf5_to_dict(key)  # Recursive call for subgroups
        return data_dict


def make_dsrl_data():
    """
    Process DSRL dataset, converting from HDF5 format to pickle format.
    Reads environment configurations from config.yaml and processes data for each environment.
    """
    # Set data directories
    hdf5_data_dir = "/home/wsj"
    data_dir = "/home/wsj/DT_WITH_CR_MAIN/data/dsrl_dataset"
    # hdf5_data_dir = "data/hdf5_dataset"  # Directory containing HDF5 files
    # data_dir = "data/dsrl_dataset"       # Directory for output pickle files
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(hdf5_data_dir):
        os.makedirs(hdf5_data_dir)
        print(f"Warning: HDF5 data directory {hdf5_data_dir} does not exist. Please place your HDF5 files there.")
        return
        
    print(f"HDF5 data directory: {hdf5_data_dir}")
    print(f"Output directory: {data_dir}")

    # Read all environments from config.yaml
    with open("utils/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    env_data_names = list(config.keys())
    
    for env_data_name in env_data_names:
        file_name = f"{env_data_name}.hdf5"
        env_name = f"{env_data_name}"
        
        pkl_file_path = os.path.join(data_dir, env_name + "safe")
        hdf5_file_path = os.path.join(hdf5_data_dir, file_name)
        
        if not os.path.exists(hdf5_file_path):
            print(f"Warning: HDF5 file not found: {hdf5_file_path}")
            continue
            
        print("Processing environment: ", env_name)
        data_dict = hdf5_to_dict(hdf5_file_path)
        N = data_dict["rewards"].shape[0]
        terminate_on_end = data_dict["terminals"][-1]
        print(f"Environment: {env_name}, Samples: {N}, Terminate on end: {terminate_on_end}")
        print(f"Data dictionary keys: {data_dict.keys()}")
        data_ = collections.defaultdict(list)
        
        episode_step = 0
        paths = []

        use_timeouts = False
        if 'timeouts' in data_dict:
            use_timeouts = True
            for i in tqdm(range(N-1)):
                done_bool = bool(data_dict["terminals"][i])    
                if use_timeouts:
                    final_timestep = data_dict['timeouts'][i]
                else:
                    final_timestep = (episode_step == 999)
                for k in ["observations", "next_observations",
                         "actions", "rewards", "terminals", "costs"]:
                    data_[k].append(data_dict[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1            

            returns = np.array([np.sum(p["rewards"]) for p in paths])
            costs = np.array([np.sum(p["costs"]) for p in paths])
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            
            print(f"Number of samples collected: {num_samples}")
            print(f"Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, "
                  f"max = {np.max(returns):.2f}, min = {np.min(returns):.2f}")
            print(f"Trajectory costs: mean = {np.mean(costs):.2f}, std = {np.std(costs):.2f}, "
                  f"max = {np.max(costs):.2f}, min = {np.min(costs):.2f}")

            with open(f"{pkl_file_path}.pkl", "wb") as f:
                pickle.dump(paths, f)


if __name__ == "__main__":
    make_dsrl_data()
