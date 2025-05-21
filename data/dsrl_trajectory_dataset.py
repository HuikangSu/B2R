import pickle
import random
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from dsrl.infos import DEFAULT_MAX_EPISODE_STEPS

def calculate_rtg_ctg(trajectories, reward_scale=1.0):
    """
    Calculate returns-to-go (RTG) and costs-to-go (CTG) for each trajectory.
    
    Args:
        trajectories (list): List of trajectory dictionaries with 'rewards' and 'costs' keys
        reward_scale (float): Scale factor for rewards
        
    Returns:
        list: Trajectories with added 'returns_to_go' and 'costs_to_go' keys
    """
    for traj in trajectories:
        # Calculate RTG with reward scaling
        rewards = traj["rewards"] / reward_scale
        traj["returns_to_go"] = discount_cumsum(rewards, gamma=1.0)
        
        # Calculate CTG
        costs = traj["costs"]
        traj["costs_to_go"] = discount_cumsum(costs, gamma=1.0)
    return trajectories

def load_and_process_trajectories_CTG_Shift(trajectories, cost_limit, cost_scale, reward_scale=1.0):
    """
    Implementation 1: Shift Strategy
    Adds a constant offset to all cost-to-go tokens along the trajectory.
    Preserves the original temporal profile of cost decay.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        cost_limit (float): Maximum allowed total cost (κ)
        cost_scale (float): Scale factor for costs
        reward_scale (float): Scale factor for rewards
        
    Returns:
        list: Processed trajectories
    """
    filtered_trajectories = [traj for traj in trajectories if np.sum(traj["costs"]) <= cost_limit]
    filtered_trajectories = calculate_rtg_ctg(filtered_trajectories, reward_scale)
    
    for traj in filtered_trajectories:
        total_cost = np.sum(traj["costs"])
        delta = cost_limit - total_cost  # Δ = κ - C(τ)
        traj["costs_to_go"] = (traj["costs_to_go"] + delta) / cost_scale  # Ĉ't = Ĉt + Δ
    return filtered_trajectories

def load_and_process_trajectories_CTG_Avg(trajectories, cost_limit, cost_scale, reward_scale=1.0):
    """
    Implementation 2: Average Strategy
    Evenly distributes the total offset across all steps of the trajectory.
    Flattens cost variation across time.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        cost_limit (float): Maximum allowed total cost (κ)
        cost_scale (float): Scale factor for costs
        reward_scale (float): Scale factor for rewards
        
    Returns:
        list: Processed trajectories
    """
    filtered_trajectories = [traj for traj in trajectories if np.sum(traj["costs"]) <= cost_limit]
    filtered_trajectories = calculate_rtg_ctg(filtered_trajectories, reward_scale)
    
    for traj in filtered_trajectories:
        total_cost = np.sum(traj["costs"])
        delta = cost_limit - total_cost  # Δ = κ - C(τ)
        H = len(traj["costs"])  # Trajectory length
        
        # Modify per-step costs: c't = ct + Δ/H
        traj["costs"] = traj["costs"] + delta / H
        
        # Recompute cost-to-go sequence using discount_cumsum
        traj["costs_to_go"] = discount_cumsum(traj["costs"], gamma=1.0) / cost_scale
            
    return filtered_trajectories

def load_and_process_trajectories_CTG_Rand(trajectories, cost_limit, max_episode_steps, cost_scale, reward_scale=1.0):
    """
    Implementation 3: Random Strategy
    Randomly reallocates the excess budget across eligible timesteps.
    For discrete environments (SafetyGym), flips randomly chosen ct=0 steps to ct=1.
    For continuous environments (MetaDrive), samples ct<κ/H steps and updates them.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        cost_limit (float): Maximum allowed total cost (κ)
        cost_scale (float): Scale factor for costs
        reward_scale (float): Scale factor for rewards
        
    Returns:
        list: Processed trajectories
    """
    filtered_trajectories = [traj for traj in trajectories if np.sum(traj["costs"]) <= cost_limit]
    filtered_trajectories = calculate_rtg_ctg(filtered_trajectories, reward_scale)
    
    for traj in filtered_trajectories:
        total_cost = np.sum(traj["costs"])
        delta = cost_limit - total_cost  # Δ = κ - C(τ)
        H = max_episode_steps
        
        # Check if this is a discrete environment (costs are 0 or 1)
        is_discrete = np.all(np.isin(traj["costs"], [0, 1]))
        
        if is_discrete:
            # For discrete environments (SafetyGym)
            remaining_delta = int(delta)  # Convert to integer for discrete steps
            while remaining_delta > 0:
                # Find all zero-cost steps
                zero_cost_indices = np.where(traj["costs"] == 0)[0]
                if len(zero_cost_indices) == 0:
                    break
                    
                # Randomly select a zero-cost step
                t = random.choice(zero_cost_indices)
                traj["costs"][t] = 1
                remaining_delta -= 1
        else:
            # For continuous environments (MetaDrive)
            remaining_delta = delta
            while remaining_delta > 0:
                # Sample a timestep with cost < κ/H
                eligible_steps = [t for t in range(H) if traj["costs"][t] < cost_limit/H]
                if not eligible_steps:
                    break
                    
                t = random.choice(eligible_steps)
                new_cost = cost_limit / H
                cost_diff = new_cost - traj["costs"][t]
                
                if cost_diff <= remaining_delta:
                    traj["costs"][t] = new_cost
                    remaining_delta -= cost_diff
                else:
                    traj["costs"][t] += remaining_delta
                    remaining_delta = 0
        
        # Recompute cost-to-go sequence using discount_cumsum
        traj["costs_to_go"] = discount_cumsum(traj["costs"], gamma=1.0) / cost_scale
            
    return filtered_trajectories

def load_and_process_trajectories_CTG_Scale(trajectories, cost_limit, cost_scale, reward_scale=1.0):
    """
    Implementation 4: Scale Strategy
    Rescales the entire CTG sequence by a multiplicative factor.
    Preserves the relative shape of the cost curve while adjusting its magnitude.
    
    Args:
        trajectories (list): List of trajectory dictionaries
        cost_limit (float): Maximum allowed total cost (κ)
        cost_scale (float): Scale factor for costs
        reward_scale (float): Scale factor for rewards
        
    Returns:
        list: Processed trajectories
    """
    filtered_trajectories = [traj for traj in trajectories if np.sum(traj["costs"]) <= cost_limit]
    filtered_trajectories = calculate_rtg_ctg(filtered_trajectories, reward_scale)
    
    for traj in filtered_trajectories:
        total_cost = np.sum(traj["costs"])
        if total_cost == 0:
            continue
            
        alpha = cost_limit / total_cost  # α = κ/C(τ)
        traj["costs_to_go"] = alpha * traj["costs_to_go"] / cost_scale  # Ĉ't = α·Ĉt
        
    return filtered_trajectories

def discount_cumsum(x, gamma):
    """
    Calculate discounted cumulative sum of a sequence.
    Used for computing returns-to-go (RTG) and costs-to-go (CTG).
    
    Args:
        x (numpy.ndarray): Input sequence
        gamma (float): Discount factor
        
    Returns:
        numpy.ndarray: Discounted cumulative sum
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum

class DSRLTrajectoryDataset(Dataset):
    """
    Dataset class for DSRL trajectories.
    Supports different CTG realignment strategies through the processing_method parameter.
    """
    def __init__(
        self,
        env_name,
        dataset_path, 
        context_len, 
        cost_limit,
        device,
        processing_method="shift"  # Options: "shift", "avg", "rand", "scale"
    ):
        self.context_len = context_len
        self.device = device
        max_test_ep_len = DEFAULT_MAX_EPISODE_STEPS[env_name]

        # Load dataset
        with open(dataset_path, "rb") as f:
            self.trajectories = pickle.load(f)
            
        # Load environment configuration
        with open("utils/B2R_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            env_config = config[env_name]
            self.cost_scale = env_config["cost_scale"]
            self.reward_scale = env_config["reward_scale"]
            
        # Process trajectories based on selected method
        if processing_method == "shift":
            self.trajectories = load_and_process_trajectories_CTG_Shift(
                self.trajectories, cost_limit, self.cost_scale, self.reward_scale)
        elif processing_method == "avg":
            self.trajectories = load_and_process_trajectories_CTG_Avg(
                self.trajectories, cost_limit, self.cost_scale, self.reward_scale)
        elif processing_method == "rand":
            self.trajectories = load_and_process_trajectories_CTG_Rand(
                self.trajectories, cost_limit, self.cost_scale, self.reward_scale)
        elif processing_method == "scale":
            self.trajectories = load_and_process_trajectories_CTG_Scale(
                self.trajectories, cost_limit, self.cost_scale, self.reward_scale)
        else:
            raise ValueError(f"Unknown processing method: {processing_method}")
        
        # Process trajectories
        self._process_trajectories(cost_limit)

    def _process_trajectories(self, cost_limit):
        """Process trajectories: calculate statistics and normalize states."""
        states, returns, costs = [], [], []
        for traj in self.trajectories:
            states.append(traj["observations"])
            returns.append(traj["rewards"].sum())
            costs.append(traj["costs"].sum())

        # Calculate state statistics
        states = np.concatenate(states, axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6

        # Normalize states
        for traj in self.trajectories:
            traj["observations"] = (traj["observations"] - self.state_mean) / self.state_std
            traj["next_observations"] = (traj["next_observations"] - self.state_mean) / self.state_std

        # Calculate return and cost statistics
        returns = np.array(returns)
        costs = np.array(costs)
        self.return_stats = [returns.max(), returns.mean(), returns.std()]
        self.cost_stats = [costs.max(), costs.mean(), costs.std()]
        
        # Print statistics
        print(f"Returns - max: {returns.max():.2f}, min: {returns.min():.2f}, "
              f"mean: {returns.mean():.2f}, std: {returns.std():.2f}")
        print(f"Costs - max: {costs.max():.2f}, min: {costs.min():.2f}, "
              f"mean: {costs.mean():.2f}, std: {costs.std():.2f}")

    def get_state_stats(self):
        """Get state statistics."""
        return self.state_mean, self.state_std
    
    def get_return_stats(self):
        """Get return statistics."""
        return self.return_stats
    
    def get_env_scale(self):
        """Get environment scales."""
        return self.reward_scale, self.cost_scale

    def __len__(self):
        """Get the number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Get a trajectory sample.
        
        Args:
            idx (int): Index of the trajectory
            
        Returns:
            tuple: (timesteps, states, next_states, actions, returns_to_go, costs_to_go, traj_mask)
        """
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # Sample random index for trajectory slicing
            si = random.randint(0, traj_len - self.context_len)
            
            # Extract trajectory segment
            states = torch.from_numpy(traj["observations"][si:si + self.context_len])
            next_states = torch.from_numpy(traj["next_observations"][si:si + self.context_len])
            actions = torch.from_numpy(traj["actions"][si:si + self.context_len])
            returns_to_go = torch.from_numpy(traj["returns_to_go"][si:si + self.context_len])
            costs_to_go = torch.from_numpy(traj["costs_to_go"][si:si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            # Handle short trajectories with padding
            padding_len = self.context_len - traj_len
            
            # Create padded tensors
            states = self._create_padded_tensor(traj["observations"], padding_len)
            next_states = self._create_padded_tensor(traj["next_observations"], padding_len)
            actions = self._create_padded_tensor(traj["actions"], padding_len)
            returns_to_go = self._create_padded_tensor(traj["returns_to_go"], padding_len)
            costs_to_go = self._create_padded_tensor(traj["costs_to_go"], padding_len)
            
            timesteps = torch.arange(start=0, end=self.context_len, step=1)
            traj_mask = torch.cat([
                torch.ones(traj_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ], dim=0)

        return (
            timesteps,
            states,
            next_states,
            actions,
            returns_to_go,
            costs_to_go,
            traj_mask,
        )

    def _create_padded_tensor(self, data, padding_len):
        """Helper method to create padded tensors."""
        tensor = torch.from_numpy(data)
        return torch.cat([
            tensor,
            torch.zeros(([padding_len] + list(tensor.shape[1:])), dtype=tensor.dtype)
        ], dim=0)
