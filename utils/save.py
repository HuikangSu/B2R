import os
import torch
from typing import List, Optional

def save_model(
    score_list: List[float],
    current_score: float,
    model_dir: str,
    env_name: str,
    episode: int,
    max_train_iters: int,
    trainer: torch.nn.Module
) -> None:
    """
    Save model checkpoint.
    Only saves the final model at the end of training.
    
    Args:
        score_list (List[float]): List of historical scores
        current_score (float): Current episode score
        model_dir (str): Directory to save models
        env_name (str): Environment name
        episode (int): Current episode number
        max_train_iters (int): Maximum number of training iterations
        trainer (torch.nn.Module): Trainer instance
    """
    # Only save the final model
    if episode == max_train_iters:
        final_model_path = os.path.join(model_dir, f"{env_name}_final.pt")
        torch.save({
            'episode': episode,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'score': current_score,
        }, final_model_path)
        print(f"Final model saved! Score: {current_score:.2f}")