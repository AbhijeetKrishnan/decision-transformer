import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
from decision_transformer.models.CategoricalMasked import CategoricalMasked


import matplotlib
import matplotlib.pyplot as plt
# Ref.: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#using-the-helper-function-code-style
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def func(x, pos):
    return f"{x:.2f}".replace("0.", ".").replace(".00", "")

class SequentialStateEmbedder(nn.Module):
    def __init__(self, sequence_length, vocabulary_size, hidden_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, vocabulary_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(vocabulary_size * sequence_length, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.shape[0], embedded.shape[1], -1)
        rnn_output, _ = self.rnn(embedded)
        final_output = self.fc(rnn_output[-1])
        return final_output

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            vocab_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            use_seq_state_embedding=False,
            use_max_log_prob=False,
            seed=0,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.use_seq_state_embedding = use_seq_state_embedding

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        if use_seq_state_embedding:
            self.embed_state = SequentialStateEmbedder(state_dim, vocab_size, hidden_size)
        else:
            self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

        self.use_max_log_prob = use_max_log_prob
        self.generator = torch.Generator(device='cuda').manual_seed(seed)

    def forward(self, states, actions, rewards, returns_to_go, action_masks, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        if self.use_seq_state_embedding:
            state_embeddings = self.embed_state(states)
        else:
            state_embeddings = self.embed_state(states.to(dtype=torch.float32)) # for compatibility with Linear layer
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        # last_action_masks = action_masks[:, -1, :].unsqueeze(1) # use the action mask of the last state in the context
        masked_action_categoricals = [CategoricalMasked(logits=action, mask=action_mask).logits for (action, action_mask) in zip(action_preds, action_masks)]
        masked_action_categoricals = torch.stack(masked_action_categoricals)

        return state_preds, masked_action_categoricals, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, action_masks, timesteps, visualize_logits=None, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        action_masks = action_masks.reshape(1, -1, self.act_dim)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            action_masks = action_masks[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.long)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            action_masks = torch.cat(
                [torch.zeros((action_masks.shape[0], self.max_length - action_masks.shape[1], self.act_dim), device=action_masks.device), action_masks],
                dim=1).to(dtype=torch.bool)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, action_masks, timesteps, attention_mask=attention_mask, **kwargs)

        last_action_pred_logits = action_preds[0, -1]
        if visualize_logits is not None:
            # visualize action logits using matplotlib
            env = visualize_logits
            rule_heads = [f'{rule.symbol.name}->{" ".join([symbol.name for symbol in rule.rhs])}' for rule in env.unwrapped.rules]
            symbols = [symbol.name for symbol in env.unwrapped.symbols]
            last_action_pred_probs = torch.nn.functional.softmax(last_action_pred_logits, dim=0).detach().cpu().numpy().reshape(env.unwrapped.num_rules, env.unwrapped.max_len)

            fig, ax = plt.subplots(1, 1)
            im, cbar = heatmap(last_action_pred_probs[:, :len(env.unwrapped.symbols)], rule_heads, symbols, ax=ax,
                               cmap="magma", cbarlabel="probability", vmin=0.0, vmax=1.0, aspect="equal")
            texts = annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=8, textcolors=("white", "black"), threshold=0.5)

            fig.tight_layout()
            plt.show()

        if self.use_max_log_prob:
            # Calculate action using max log prob
            action = last_action_pred_logits.max(0, keepdim=True)[1][0]
        else:
            # Sample action from distribution
            last_action_pred_probs = torch.nn.functional.softmax(last_action_pred_logits, dim=0)
            action = torch.multinomial(last_action_pred_probs, num_samples=1, generator=self.generator).squeeze()

        return action
