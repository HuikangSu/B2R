import torch
import torch.nn as nn
import numpy as np
from model.net import TransformerBlock, BaseActor

class B2R(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim, 
        n_blocks, 
        h_dim, 
        context_len,
        n_heads, 
        drop_p, 
        init_temperature,
        target_entropy,
        use_rope,
        device,
        max_timestep=4096,
        dt_mask=False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.use_rope = use_rope
        self.device = device

        ### transformer blocks
        self.num_inputs = 4
        input_seq_len = self.num_inputs * context_len
        blocks = [
            TransformerBlock(
                h_dim, 
                input_seq_len, 
                n_heads, 
                drop_p,
                self.num_inputs,
                use_rope=self.use_rope,
                mgdt=False,
                dt_mask=dt_mask,
                att_mask=None,
                device=device,
            ) 
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)
        self.embed_ctg = nn.Linear(1, h_dim)

        ### prediction heads
        self.predict_rtg = nn.Linear(h_dim, 1)
        # stochastic action
        self.predict_action = BaseActor(h_dim, self.act_dim)
        self.predict_state = nn.Linear(h_dim, state_dim)
        self.predict_ctg = nn.Linear(h_dim, 1)

        # For entropy
        self.log_temperature = torch.tensor(np.log(init_temperature))
        self.log_temperature.requires_grad = True
        self.target_entropy = target_entropy

    def temperature(self):
        return self.log_temperature.exp()

    def forward(
        self, 
        timesteps, 
        states, 
        costs_to_go,
        returns_to_go,
        actions, 
    ):

        B, T, _ = states.shape
        #B = batch_size,T = timesteps

        if not self.use_rope:
            time_embeddings = self.embed_timestep(timesteps)
            # time embeddings are treated similar to positional embeddings
            state_embeddings = self.embed_state(states) + time_embeddings
            action_embeddings = self.embed_action(actions) + time_embeddings
            rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
            ctg_embeddings = self.embed_ctg(costs_to_go) + time_embeddings
        else:
            state_embeddings = self.embed_state(states)
            action_embeddings = self.embed_action(actions)
            rtg_embeddings = self.embed_rtg(returns_to_go)
            ctg_embeddings = self.embed_ctg(costs_to_go)
        h = (
            torch.stack(
                (
                    state_embeddings,
                    ctg_embeddings,
                    rtg_embeddings, 
                    action_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t, R_t
        # h[:, 2, t] is conditioned on the input sequence s_0, R_0, a_0 ... s_t, R_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (s_t, R_t, a_t) in sequence.
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        ctg_preds = self.predict_ctg(h[:, 0])
        rtg_preds  = self.predict_rtg(h[:, 1])        # predict rtg given s
        action_dist_preds = self.predict_action(h[:, 2])  # predict action given s, R
        state_preds = self.predict_state(h[:, 3])         # predict next state given s, R, a

        return (
            ctg_preds,
            rtg_preds,
            action_dist_preds, 
            state_preds, 
        )

