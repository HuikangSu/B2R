import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from model.B2R import B2R
from utils.lamb import Lamb
import wandb

class B2RTrainer:
    """
    Trainer for B2R (Behavior Transformer with Return) model.
    
    This trainer implements the training logic for the B2R model, including:
    - Model initialization and optimization
    - Training step execution
    - Loss computation
    - Temperature parameter optimization
    """
    
    def __init__(
        self, 
        state_dim: int,
        act_dim: int,
        device: torch.device,
        variant: Dict
    ):
        """
        Initialize the B2R trainer.
        
        Args:
            state_dim (int): Dimension of state space
            act_dim (int): Dimension of action space
            device (torch.device): Device to run training on
            variant (Dict): Configuration dictionary containing:
                - grad_norm (float): Gradient clipping norm
                - tau (float): Temperature parameter
                - context_len (int): Context length for transformer
                - use_wandb (bool): Whether to use wandb logging
                - use_rope (bool): Whether to use rotary position embeddings
                - n_blocks (int): Number of transformer blocks
                - embed_dim (int): Embedding dimension
                - n_heads (int): Number of attention heads
                - dropout_p (float): Dropout probability
                - init_temperature (float): Initial temperature value
                - lr (float): Learning rate
                - wd (float): Weight decay
                - warmup_steps (int): Number of warmup steps
        """
        super().__init__()
        
        # Store basic parameters
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device
        
        # Store variant parameters
        self.grad_norm = variant["grad_norm"]
        self.tau = variant["tau"]
        self.context_len = variant["context_len"]
        self.use_wandb = variant["use_wandb"]
        self.use_rope = variant["use_rope"]
        
        # Initialize model
        self.model = B2R(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=variant["n_blocks"],
            h_dim=variant["embed_dim"],
            context_len=variant["context_len"],
            n_heads=variant["n_heads"],
            drop_p=variant["dropout_p"],
            init_temperature=variant["init_temperature"],
            target_entropy=-self.act_dim,
            use_rope=self.use_rope,
            device=variant["device"],
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["lr"],
            weight_decay=variant["wd"],
            eps=1e-8,
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/variant["warmup_steps"], 1)
        )
        
        # Initialize temperature optimizer
        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )
    
    def train_step(
        self,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        costs_to_go: torch.Tensor,
        returns_to_go: torch.Tensor,
        traj_mask: torch.Tensor,
    ) -> float:
        """
        Execute one training step.
        
        Args:
            timesteps (torch.Tensor): Batch of timesteps (B x T)
            states (torch.Tensor): Batch of states (B x T x state_dim)
            next_states (torch.Tensor): Batch of next states (B x T x state_dim)
            actions (torch.Tensor): Batch of actions (B x T x act_dim)
            costs_to_go (torch.Tensor): Batch of costs-to-go (B x T)
            returns_to_go (torch.Tensor): Batch of returns-to-go (B x T)
            traj_mask (torch.Tensor): Batch of trajectory masks (B x T)
            
        Returns:
            float: Training loss value
        """
        self.model.train()
        
        # Move data to device
        timesteps = timesteps.to(self.device)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        costs_to_go = costs_to_go.to(self.device).unsqueeze(dim=-1)
        returns_to_go = returns_to_go.to(self.device).unsqueeze(dim=-1)
        traj_mask = traj_mask.to(self.device)
        
        # Forward pass
        (_, _, actions_dist_preds, _) = self.model.forward(
            timesteps=timesteps,
            states=states,
            costs_to_go=costs_to_go,
            returns_to_go=returns_to_go,
            actions=actions,
        )
        
        # Compute action loss
        actions_target = torch.clone(actions)
        log_likelihood = actions_dist_preds.log_prob(actions_target).sum(axis=2)[traj_mask > 0].mean()
        entropy = actions_dist_preds.entropy().sum(axis=2).mean()
        action_loss = -(log_likelihood + self.model.temperature().detach() * entropy)
        
        # Total loss
        loss = action_loss
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                "training/action_loss": action_loss,
            })
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        
        # Temperature optimization
        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return loss.detach().cpu().item()