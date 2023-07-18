import numpy as np
import torch
import torch.nn as nn

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.CategoricalMasked import CategoricalMasked


class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        # TODO: replace initial layer with torch.nn.Embedding layer
        # and simultaneously one-hot encode observation dimension (or keep indices, whatever the layer specifies)
        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, action_masks, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat last K states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        latest_action_masks = action_masks[:, -1, :].unsqueeze(1) # use latest state's action mask
        masked_action_categoricals = [CategoricalMasked(logits=action, mask=latest_action_mask).probs for (action, latest_action_mask) in zip(actions, latest_action_masks)]
        masked_action_categoricals = torch.stack(masked_action_categoricals)

        return None, masked_action_categoricals, None

    def get_action(self, states, actions, rewards, action_masks, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        action_masks = action_masks.reshape(1, -1, self.act_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, action_masks, **kwargs)
        action = actions[0, -1].max(0, keepdim=True)[1][0] # get index of max log-probability
        return action