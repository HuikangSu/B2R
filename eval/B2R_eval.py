import torch
import numpy as np
from typing import Tuple, Dict, Any
from tqdm import tqdm

class B2REvaluator:
    """
    Evaluator for B2R (Behavior Transformer with Return) model.
    
    This evaluator implements the evaluation logic for the B2R model, including:
    - Environment interaction
    - Model inference
    - Performance metrics calculation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        context_len: int,
        env: Any,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        reward_scale: float,
        cost_scale: float,
        target_rtg: float,
        cost_limit: float,
        num_eval_ep: int = 10,
        max_test_ep_len: int = 200,
    ):
        """
        Initialize the B2R evaluator.
        
        Args:
            model (torch.nn.Module): B2R model to evaluate
            device (torch.device): Device to run evaluation on
            context_len (int): Context length for transformer
            env (Any): Environment to evaluate on
            state_mean (np.ndarray): Mean of state normalization
            state_std (np.ndarray): Standard deviation of state normalization
            reward_scale (float): Scale factor for rewards
            cost_scale (float): Scale factor for costs
            target_rtg (float): Target return-to-go
            cost_limit (float): Cost limit for evaluation
            num_eval_ep (int): Number of evaluation episodes
            max_test_ep_len (int): Maximum episode length
        """
        self.model = model
        self.device = device
        self.context_len = context_len
        self.env = env
        self.state_mean = torch.from_numpy(state_mean).to(device)
        self.state_std = torch.from_numpy(state_std).to(device)
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.target_rtg = target_rtg
        self.cost_limit = cost_limit
        self.num_eval_ep = num_eval_ep
        self.max_test_ep_len = max_test_ep_len
        
        # Get environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Initialize timesteps tensor
        self.timesteps = torch.arange(
            start=0, 
            end=max_test_ep_len, 
            step=1
        ).repeat(1, 1).to(device)
    
    def _create_placeholders(self) -> Dict[str, torch.Tensor]:
        """
        Create zero tensors for storing episode data.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing zero tensors for states,
                costs_to_go, returns_to_go, and actions
        """
        return {
            "states": torch.zeros(
                (1, self.max_test_ep_len, self.state_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "costs_to_go": torch.zeros(
                (1, self.max_test_ep_len, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            "returns_to_go": torch.zeros(
                (1, self.max_test_ep_len, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            "actions": torch.zeros(
                (1, self.max_test_ep_len, self.act_dim),
                dtype=torch.float32,
                device=self.device,
            ),
        }
    
    def _get_action(
        self,
        t: int,
        placeholders: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get action from model for current timestep.
        
        Args:
            t (int): Current timestep
            placeholders (Dict[str, torch.Tensor]): Dictionary containing episode data
            
        Returns:
            torch.Tensor: Action tensor
        """
        if t < self.context_len:
            # Use initial context
            _, _, act_dist_preds, _ = self.model.forward(
                self.timesteps[:, :self.context_len],
                placeholders["states"][:, :self.context_len],
                placeholders["costs_to_go"][:, :self.context_len],
                placeholders["returns_to_go"][:, :self.context_len],
                placeholders["actions"][:, :self.context_len],
            )
            return act_dist_preds.mean.reshape(1, -1, self.act_dim)[0, t].detach()
        else:
            # Use sliding window context
            _, _, act_dist_preds, _ = self.model.forward(
                self.timesteps[:, t - self.context_len + 1 : t + 1],
                placeholders["states"][:, t - self.context_len + 1 : t + 1],
                placeholders["costs_to_go"][:, t - self.context_len + 1 : t + 1],
                placeholders["returns_to_go"][:, t - self.context_len + 1 : t + 1],
                placeholders["actions"][:, t - self.context_len + 1 : t + 1],
            )
            return act_dist_preds.mean.reshape(1, -1, self.act_dim)[0, -1].detach()
    
    def evaluate(self) -> Tuple[float, float, float, float, float]:
        """
        Evaluate the model for multiple episodes.
        
        Returns:
            Tuple[float, float, float, float, float]: Mean return, return std,
                mean length, mean cost return, cost return std
        """
        self.model.eval()
        returns = []
        lengths = []
        cost_returns = []
        
        with torch.no_grad():
            for _ in tqdm(range(self.num_eval_ep), desc="Evaluating", leave=False):
                # Initialize episode
                placeholders = self._create_placeholders()
                state, _ = self.env.reset()
                episode_return = 0
                episode_length = 0
                episode_cost = 0
                
                for t in range(self.max_test_ep_len):
                    # Update state and normalize
                    placeholders["states"][0, t] = (torch.from_numpy(state).to(self.device) - self.state_mean) / self.state_std
                    
                    # Update costs-to-go and returns-to-go
                    if episode_cost < self.cost_limit:
                        placeholders["costs_to_go"][0, t] = (self.cost_limit - episode_cost) / self.cost_scale
                    else:
                        placeholders["costs_to_go"][0, t] = 0 / self.cost_scale
                    placeholders["returns_to_go"][0, t] = (self.target_rtg - episode_return) / self.reward_scale
                    
                    # Get action from model
                    action = self._get_action(t, placeholders)
                    placeholders["actions"][0, t] = action
                    
                    # Environment step
                    state, reward, done, truncated, info = self.env.step(action.cpu().numpy())
                    
                    # Update episode statistics
                    episode_return += reward
                    episode_length += 1
                    episode_cost += info["cost"]
                    
                    # Check termination
                    if done or truncated:
                        returns.append(episode_return)
                        lengths.append(episode_length)
                        cost_returns.append(episode_cost)
                        break
        
        # Calculate statistics
        returns_array = np.array(returns)
        lengths_array = np.array(lengths)
        cost_returns_array = np.array(cost_returns)
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        mean_length = np.mean(lengths_array)
        mean_cost_return = np.mean(cost_returns_array)
        std_cost_return = np.std(cost_returns_array)
        
        # Print results
        print(f"Mean Return: {mean_return:.2f}")
        print(f"Mean Length: {mean_length:.2f}")
        print(f"Mean Cost Return: {mean_cost_return:.2f}")
        
        return mean_return, std_return, mean_length, mean_cost_return, std_cost_return
