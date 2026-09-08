from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2-Vpi world model.
	Replaces the Q(s,a) critic with an on-policy state-value function V_pol trained via
	Bellman bootstrap on policy-sampled transitions: y = r(z,~a) + gamma*V_pol(f(z,~a)), ~a~pi.
	The policy is updated via expanded model-based gradient: d/dtheta[r(z,pi(z)) + gamma*V_pol(f(z,pi(z)))].
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._V = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._V.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		self._detach_V_params = TensorDictParams(self._V.params.data, no_convert=True)
		self._target_V_params = TensorDictParams(self._V.params.data.clone(), no_convert=True)

		with self._detach_V_params.data.to("meta").to_module(self._V.module):
			self._detach_V = deepcopy(self._V)
			self._target_V = deepcopy(self._V)

		delattr(self._detach_V, "params")
		self._detach_V.__dict__["params"] = self._detach_V_params
		delattr(self._target_V, "params")
		self._target_V.__dict__["params"] = self._target_V_params

	def __repr__(self):
		repr = 'TD-MPC2-Vpi World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy', 'V_pol']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._V]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		super().train(mode)
		self._target_V.train(False)
		return self

	def soft_update_target_V(self):
		self._target_V_params.lerp_(self._detach_V_params, self.cfg.tau)

	def task_emb(self, x, task):
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def termination(self, z, task, unnormalized=False):
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))

	def pi(self, z, task):
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask:
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else:
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		pre_tanh_action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, pre_tanh_action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"pre_tanh_action": pre_tanh_action,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def V(self, z, task, return_type='min', target=False, detach=False):
		"""Predict state value V_pol(z)."""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		if target:
			vnet = self._target_V
		elif detach:
			vnet = self._detach_V
		else:
			vnet = self._V

		out = vnet(z)
		if return_type == 'all':
			return out

		vidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		V = math.two_hot_inv(out[vidx], self.cfg)
		if return_type == 'min':
			return V.min(0).values
		return V.sum(0) / 2
