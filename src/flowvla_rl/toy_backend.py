"""Small CPU-only integration fixture. Never represents LIBERO task performance."""
from types import SimpleNamespace

import torch
from torch import nn

from flowvla_rl.smolvla_sde import prepare_policy_prefix, sample_sde_chunk
from flowvla_rl.training import StoredChunk


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(5, 5)
        self.vlm.requires_grad_(False)
        self.expert = nn.Linear(5, 5)

    def forward(self, *, inputs_embeds, **kwargs):
        return None, self.vlm(inputs_embeds[0])


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_steps=10, chunk_size=4, max_action_dim=5, use_cache=True)
        self.vlm_with_expert = TinyBackbone()
        self.state_proj = nn.Linear(5, 5)
        self.state_proj.requires_grad_(False)
        self.action_out_proj = nn.Linear(5, 5)

    def embed_prefix(self, images, image_masks, language_tokens, language_masks, state):
        masks = torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device)
        return self.state_proj(state)[:, None], masks, torch.zeros_like(masks)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
        hidden = torch.tanh(self.vlm_with_expert.expert(x_t) + past_key_values + timestep[:, None, None])
        return self.action_out_proj(hidden) * 0.1


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyFlow()
        self.config = self.model.config

    def _prepare_batch(self, batch):
        return batch

    def prepare_images(self, batch):
        return [], []

    def prepare_state(self, batch):
        return batch["state"]


def make_chunks(policy, seed=10):
    device = next(policy.parameters()).device
    # fork_rng makes this fixture safe to call within RNG independence tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        chunks = []
        for episode, count in enumerate((1, 2, 1, 2)):
            for chunk_id in range(count):
                conditioning = {"state": torch.randn(1, 5, device=device),
                                "observation.language.tokens": torch.zeros(1, 1, dtype=torch.long, device=device),
                                "observation.language.attention_mask": torch.ones(1, 1, dtype=torch.bool, device=device)}
                with torch.no_grad():
                    masks, cache = prepare_policy_prefix(policy, conditioning)
                    trajectory = sample_sde_chunk(policy.model, masks, cache, action_dim=3, eta=0.4)
                positions = torch.tensor([[True, True, True, chunk_id == 0]], device=device)
                chunks.append(StoredChunk(episode, conditioning, trajectory, positions,
                                          0.5 if episode % 2 == 0 else -0.5, chunk_id))
        return chunks
