import os
import numpy as np
import collections
import pickle
import h5py
import requests
from tqdm import tqdm
import yaml

# ==========================================
# 1. 数据集下载链接配置
# ==========================================
DATASET_URLS = {
    # bullet_safety_gym
    "AntCircle": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntCircle-v0-200-5728.hdf5",
    "AntRun": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntRun-v0-150-1816.hdf5",
    "BallCircle": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyBallCircle-v0-80-886.hdf5",
    "BallRun": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyBallRun-v0-80-940.hdf5",
    "CarCircle": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle-v0-100-1450.hdf5",
    "CarRun": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarRun-v0-40-651.hdf5",
    "DroneCircle": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyDroneCircle-v0-100-1923.hdf5",
    "DroneRun": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyDroneRun-v0-140-1990.hdf5",
    # safety_gymnasium: Car
    "CarButton1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarButton1Gymnasium-v0-250-2656.hdf5",
    "CarButton2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarButton2Gymnasium-v0-300-3755.hdf5",
    "CarCircle1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle1Gymnasium-v0-250-1271.hdf5",
    "CarCircle2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarCircle2Gymnasium-v0-400-940.hdf5",
    "CarGoal1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarGoal1Gymnasium-v0-120-1671.hdf5",
    "CarGoal2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarGoal2Gymnasium-v0-200-4105.hdf5",
    "CarPush1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarPush1Gymnasium-v0-200-2871.hdf5",
    "CarPush2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyCarPush2Gymnasium-v0-250-4407.hdf5",
    # safety_gymnasium: Point
    "PointButton1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointButton1Gymnasium-v0-200-2268.hdf5",
    "PointButton2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointButton2Gymnasium-v0-250-3288.hdf5",
    "PointCircle1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointCircle1Gymnasium-v0-200-1098.hdf5",
    "PointCircle2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointCircle2Gymnasium-v0-300-895.hdf5",
    "PointGoal1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5",
    "PointGoal2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointGoal2Gymnasium-v0-200-3442.hdf5",
    "PointPush1": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointPush1Gymnasium-v0-150-2379.hdf5",
    "PointPush2": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyPointPush2Gymnasium-v0-200-3242.hdf5",
    # safety_gymnasium: velocity
    "AntVelocity": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyAntVelocityGymnasium-v1-250-2249.hdf5",
    "HalfCheetahVelocity": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyHalfCheetahVelocityGymnasium-v1-250-2495.hdf5",
    "HopperVelocity": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyHopperVelocityGymnasium-v1-250-2240.hdf5",
    "SwimmerVelocity": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetySwimmerVelocityGymnasium-v1-200-1686.hdf5",
    "Walker2dVelocity": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafetyWalker2dVelocityGymnasium-v1-300-2729.hdf5",
    # safe_metadrive
    "easysparse": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easysparse-v0-85-1000.hdf5",
    "easymean": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easymean-v0-85-1000.hdf5",
    "easydense": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-easydense-v0-85-1000.hdf5",
    "mediumsparse": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediumsparse-v0-50-1000.hdf5",
    "mediummean": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediummean-v0-50-1000.hdf5",
    "mediumdense": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-mediumdense-v0-50-1000.hdf5",
    "hardsparse": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-hardsparse-v0-85-1000.hdf5",
    "hardmean": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-hardmean-v0-85-1000.hdf5",
    "harddense": "https://huggingface.co/datasets/YYY-45/DSRL/resolve/main/SafeMetaDrive-harddense-v0-85-1000.hdf5"
}

# ==========================================
# 2. 辅助函数
# ==========================================

def download_file(url, filename):
    """
    Helper function to download a file from a URL with a progress bar.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"Downloading {filename}...")
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f"Successfully downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Remove partial file
        return False

def hdf5_to_dict(hdf5_file_path):
    """
    Convert HDF5 file to Python dictionary.
    """
    with h5py.File(hdf5_file_path, 'r') as file:
        data_dict = {}
        for key in file.keys():
            if isinstance(file[key], h5py.Dataset):
                data_dict[key] = file[key][:]
            elif isinstance(file[key], h5py.Group):
                data_dict[key] = hdf5_to_dict(key)  # Recursive call for subgroups
        return data_dict

# ==========================================
# 3. 主逻辑
# ==========================================

def make_dsrl_data():
    """
    Process DSRL dataset: Download (if needed) -> Convert HDF5 to Pickle.
    """
    # Set data directories
    hdf5_data_dir = "data/hdf5_dataset"  # Directory containing HDF5 files
    data_dir = "data/dsrl_dataset"       # Directory for output pickle files
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(hdf5_data_dir, exist_ok=True)
        
    print(f"HDF5 data directory: {hdf5_data_dir}")
    print(f"Output directory: {data_dir}")

    # Read all environments from config.yaml
    config_path = "utils/B2R_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    env_data_names = list(config.keys())
    
    for env_data_name in env_data_names:
        # Define paths
        # Note: We save the downloaded file as "EnvName.hdf5" to match the script's logic,
        # even if the source URL has a longer name.
        file_name = f"{env_data_name}.hdf5"
        env_name = f"{env_data_name}"
        
        hdf5_file_path = os.path.join(hdf5_data_dir, file_name)
        pkl_file_path = os.path.join(data_dir, env_name) # Will append .pkl later

        # --- STEP 1: Check and Download ---
        if not os.path.exists(hdf5_file_path):
            print(f"File not found locally: {file_name}")
            if env_data_name in DATASET_URLS:
                url = DATASET_URLS[env_data_name]
                print(f"Found URL for {env_data_name}, starting download...")
                success = download_file(url, hdf5_file_path)
                if not success:
                    print(f"Skipping {env_data_name} due to download failure.")
                    continue
            else:
                print(f"Warning: No URL found for {env_data_name} in DATASET_URLS. Skipping.")
                continue
        else:
            print(f"File exists locally: {file_name}")

        # --- STEP 2: Process and Convert ---
        print(f"Processing environment: {env_name}")
        try:
            data_dict = hdf5_to_dict(hdf5_file_path)
        except Exception as e:
            print(f"Failed to read HDF5 file {hdf5_file_path}: {e}")
            continue

        N = data_dict["rewards"].shape[0]
        # Check if 'terminals' exists, handle edge cases
        if "terminals" in data_dict:
            terminate_on_end = data_dict["terminals"][-1]
        else:
            terminate_on_end = False # Default or handle accordingly
            
        print(f"Samples: {N}, Terminate on end: {terminate_on_end}")
        
        data_ = collections.defaultdict(list)
        episode_step = 0
        paths = []

        use_timeouts = False
        if 'timeouts' in data_dict:
            use_timeouts = True
        
        # Using tqdm for processing progress
        print("Converting trajectories...")
        for i in tqdm(range(N-1), desc=f"Parsing {env_name}"):
            # Handle boolean conversion safely
            done_bool = bool(data_dict["terminals"][i]) if "terminals" in data_dict else False
            
            if use_timeouts:
                final_timestep = data_dict['timeouts'][i]
            else:
                final_timestep = (episode_step == 999)
            
            for k in ["observations", "next_observations", "actions", "rewards", "terminals", "costs"]:
                if k in data_dict:
                    data_[k].append(data_dict[k][i])
                else:
                    # Handle missing keys if necessary, or just skip
                    pass

            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            
            episode_step += 1            

        if not paths:
            print(f"Warning: No trajectories extracted for {env_name}.")
            continue

        returns = np.array([np.sum(p["rewards"]) for p in paths])
        costs = np.array([np.sum(p["costs"]) for p in paths])
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        
        print(f"Number of samples collected: {num_samples}")
        print(f"Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, "
                f"max = {np.max(returns):.2f}, min = {np.min(returns):.2f}")
        print(f"Trajectory costs: mean = {np.mean(costs):.2f}, std = {np.std(costs):.2f}, "
                f"max = {np.max(costs):.2f}, min = {np.min(costs):.2f}")

        # Check if output file already exists to avoid unnecessary write (optional)
        output_file = f"{pkl_file_path}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(paths, f)
        print(f"Saved to {output_file}\n" + "-"*40)

if __name__ == "__main__":
    make_dsrl_data()