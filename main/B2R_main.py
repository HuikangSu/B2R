import argparse
import os
import random
from datetime import datetime
import time
from typing import Dict, Any, Tuple, Iterator
from tqdm import tqdm
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
import yaml

from data.dsrl_trajectory_dataset import DSRLTrajectoryDataset
from trainer.B2R_trainer import B2RTrainer
from eval.B2R_eval import B2REvaluator
from utils.save import save_model
from main.logger import setup_logger
import dsrl

# 设置日志记录器
logger = setup_logger("B2R")

def load_config() -> Dict:
    """
    Load configuration from YAML file.
    
    Returns:
        Dict: Configuration dictionary
    """
    logger.info("Loading configuration from YAML file")
    with open("utils/B2R_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
    return config

def get_next_batch(data_iter: Iterator, data_loader: DataLoader) -> Tuple[torch.Tensor, ...]:
    """
    Get next batch of data from data loader.
    
    Args:
        data_iter (Iterator): Data iterator
        data_loader (DataLoader): Data loader
        
    Returns:
        Tuple[torch.Tensor, ...]: Batch of data
    """
    try:
        return next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        return next(data_iter)

def setup_environment(env_name: str, cost_limit: float) -> Any:
    """
    Setup training environment.
    
    Args:
        env_name (str): Name of the environment
        cost_limit (float): Cost limit for the environment
        
    Returns:
        Any: Environment instance
    """
    config = load_config()
    env_type = config[env_name]["env_type"]
    dsrl_env = config[env_name]["env_id"]
    
    if env_type == "gymnasium":
        import gymnasium as gym
    else:
        import gym

    env = gym.make(dsrl_env)
    env.set_target_cost(cost_limit)
    return env

def setup_dataset(
    env_name: str,
    dataset_path: str,
    context_len: int,
    cost_limit: float,
    device: torch.device,
    variant: Dict
) -> Tuple[DSRLTrajectoryDataset, DataLoader]:
    """
    Setup dataset and data loader.
    
    Args:
        env_name (str): Name of the environment
        dataset_path (str): Path to dataset file
        context_len (int): Context length for transformer
        cost_limit (float): Cost limit for dataset
        device (torch.device): Device to load data on
        variant (Dict): Configuration dictionary
        
    Returns:
        Tuple[DSRLTrajectoryDataset, DataLoader]: Dataset and data loader
    """
    traj_dataset = DSRLTrajectoryDataset(
        env_name, dataset_path, context_len, cost_limit, device, variant["processing_method"]
    )
    sampler = RandomSampler(traj_dataset, replacement=True)
    
    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=variant["batch_size"],
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
    
    return traj_dataset, traj_data_loader

def setup_trainer(
    state_dim: int,
    act_dim: int,
    device: torch.device,
    variant: Dict
) -> B2RTrainer:
    """
    Setup trainer.
    
    Args:
        state_dim (int): Dimension of state space
        act_dim (int): Dimension of action space
        device (torch.device): Device to run training on
        variant (Dict): Configuration dictionary
        
    Returns:
        B2RTrainer: Trainer instance
    """
    return B2RTrainer(
        state_dim=state_dim,
        act_dim=act_dim,
        device=device,
        variant=variant
    )

def create_evaluator(
    env: Any,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    reward_scale: float,
    cost_scale: float,
    target_rtg: float,
    variant: Dict
) -> callable:
    """
    Create evaluator function.
    
    Args:
        env (Any): Environment instance
        state_mean (np.ndarray): Mean of state normalization
        state_std (np.ndarray): Standard deviation of state normalization
        reward_scale (float): Scale factor for rewards
        cost_scale (float): Scale factor for costs
        target_rtg (float): Target return-to-go
        variant (Dict): Configuration dictionary
        
    Returns:
        callable: Evaluator function
    """
    def evaluator(model: torch.nn.Module) -> Tuple[float, float]:
        """
        Evaluate model performance.
        
        Args:
            model (torch.nn.Module): Model to evaluate
            
        Returns:
            Tuple[float, float]: Normalized score and cost
        """
        evaluator = B2REvaluator(
            model=model,
            device=variant["device"],
            context_len=variant["context_len"],
            env=env,
            state_mean=state_mean,
            state_std=state_std,
            reward_scale=reward_scale,
            cost_scale=cost_scale,
            target_rtg=target_rtg,
            cost_limit=variant["cost_limit"],
            num_eval_ep=variant["num_eval_ep"],
            max_test_ep_len=variant["max_eval_ep_len"],
        )
        return_mean, _, _, cost_return_mean, _ = evaluator.evaluate()
        return env.get_normalized_score(return_mean, cost_return_mean)
    
    return evaluator

def experiment(variant: Dict) -> None:
    """
    Main experiment function.
    
    Args:
        variant (Dict): Configuration dictionary
    """
    # Basic setup
    env_name = variant["env"]
    seed = variant["seed"]
    cost_limit = variant["cost_limit"]
    
    logger.info(f"Starting experiment with environment: {env_name}, seed: {seed}, cost limit: {cost_limit}")
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seeds set successfully")
    
    # Setup environment
    logger.info("Setting up environment")
    env = setup_environment(env_name, cost_limit)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    logger.info(f"Environment setup complete. State dim: {state_dim}, Action dim: {act_dim}")
    
    # Setup dataset
    logger.info("Setting up dataset")
    dataset_path = os.path.join(variant["dataset_dir"], f"{env_name}.pkl")
    traj_dataset, traj_data_loader = setup_dataset(
        env_name, dataset_path, variant["context_len"], cost_limit, variant["device"], variant
    )
    logger.info("Dataset setup complete")
    
    # Setup trainer
    logger.info("Setting up trainer")
    trainer = setup_trainer(state_dim, act_dim, variant["device"], variant)
    logger.info("Trainer setup complete")
    
    # Setup evaluator
    logger.info("Setting up evaluator")
    state_mean, state_std = traj_dataset.get_state_stats()
    reward_scale, cost_scale = traj_dataset.get_env_scale()
    target_rtg = traj_dataset.get_return_stats()[0]  # max return
    evaluator = create_evaluator(
        env, state_mean, state_std, reward_scale, cost_scale, target_rtg, variant
    )
    logger.info("Evaluator setup complete")
    
    # Training loop
    logger.info("Starting training loop")
    normalized_dsrl_score_list = []
    normalized_dsrl_cost_list = []
    
    for iter_idx in range(variant["max_train_iters"]):
        logger.info(f"Starting iteration {iter_idx + 1}/{variant['max_train_iters']}")
        
        # Training
        data_iter = iter(traj_data_loader)
        for update_idx in tqdm(range(variant["num_updates_per_iter"]), desc=f"Training iteration {iter_idx + 1}"):
            batch = get_next_batch(data_iter, traj_data_loader)
            loss = trainer.train_iteration(batch)
            
            if update_idx % 100 == 0:
                logger.info(f"Update {update_idx}/{variant['num_updates_per_iter']}, Loss: {loss:.4f}")
        
        # Evaluation
        logger.info("Starting evaluation")
        normalized_dsrl_score, normalized_dsrl_cost = evaluator(trainer.model)
        normalized_dsrl_score_list.append(normalized_dsrl_score)
        normalized_dsrl_cost_list.append(normalized_dsrl_cost)
        
        logger.info(f"Iteration {iter_idx + 1} results:")
        logger.info(f"Normalized DSRL Score: {normalized_dsrl_score:.4f}")
        logger.info(f"Normalized DSRL Cost: {normalized_dsrl_cost:.4f}")
        
        # Save model
        if (iter_idx + 1) % 10 == 0:
            logger.info("Saving model checkpoint")
            save_model(trainer.model, f"checkpoints/model_iter_{iter_idx + 1}.pt")
    
    # Print final results
    logger.info("Training completed. Final results:")
    logger.info(f"Score list: {normalized_dsrl_score_list}")
    logger.info(f"Mean score: {np.mean(normalized_dsrl_score_list):.2f}")
    logger.info(f"Cost list: {normalized_dsrl_cost_list}")
    logger.info(f"Mean cost: {np.mean(normalized_dsrl_cost_list):.2f}")
    logger.info(f"Best score: {max(normalized_dsrl_score_list):.2f} (Episode {normalized_dsrl_score_list.index(max(normalized_dsrl_score_list)) + 1})")
    
    if variant["use_wandb"]:
        wandb.log({
            "evaluation/max_score": max(normalized_dsrl_score_list),
            "evaluation/last_score": normalized_dsrl_score_list[-1]
        })
    
    # Print end information
    logger.info("=" * 60)
    logger.info("Training completed!")
    end_time = datetime.now().replace(microsecond=0)
    logger.info(f"End time: {end_time.strftime('%y-%m-%d-%H-%M-%S')}")
    logger.info("=" * 60)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--n_blocks", type=int, default=3,
                      help="Number of transformer blocks")
    parser.add_argument("--embed_dim", type=int, default=128,
                      help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                      help="Number of attention heads")
    parser.add_argument("--dropout_p", type=float, default=0.1,
                      help="Dropout probability")
    parser.add_argument("--grad_norm", type=float, default=0.25,
                      help="Gradient clipping norm")
    parser.add_argument("--tau", type=float, default=0.001,
                      help="Temperature parameter")
    parser.add_argument("--use_rope", type=lambda x: (str(x).lower() == 'true'), default=True,
                      help="Whether to use rotary position embeddings")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4,
                      help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=5000,
                      help="Number of warmup steps")
    parser.add_argument("--max_train_iters", type=int, default=20,
                      help="Maximum number of training iterations")
    parser.add_argument("--num_updates_per_iter", type=int, default=5000,
                      help="Number of updates per iteration")
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="Batch size")
    parser.add_argument("--init_temperature", type=float, default=0.1,
                      help="Initial temperature value")
    
    # Environment parameters
    parser.add_argument("--env", type=str, default="CarCircle2",
                      help="Environment name")
    parser.add_argument("--cost_limit", type=float, default=20,
                      help="Cost limit")
    parser.add_argument("--context_len", type=int, default=3,
                      help="Context length")
    parser.add_argument("--num_eval_ep", type=int, default=20,
                      help="Number of evaluation episodes")
    parser.add_argument("--max_eval_ep_len", type=int, default=1000,
                      help="Maximum evaluation episode length")
    parser.add_argument("--dataset_dir", type=str, default="/home/wsj/DT_WITH_CR_MAIN/data/dsrl_dataset",
                      help="Dataset directory")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=1,
                      help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run on")
    parser.add_argument("--use_wandb", type=lambda x: (str(x).lower() == 'true'), default=False,
                      help="Whether to use wandb logging")
    parser.add_argument("--processing_method", type=str, default="shift",
                      help="Trajectory processing method")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    variant = vars(args)
    experiment(variant)
