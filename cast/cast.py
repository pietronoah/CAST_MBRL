import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class CAST(torch.nn.Module):
	"""
	CAST agent, built on top of TD-MPC2.
	Replaces the Q(s,a) critic with V_pol: an on-policy state-value function trained via
	Bellman bootstrap on fresh policy samples: y = r(z,~a) + gamma*V_pol(f(z,~a)), ~a~pi.
	Policy is updated via expanded model-based gradient through reward + dynamics.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._V.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20)
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		plan = torch.compile(self._plan, mode="reduce-overhead") if self.cfg.compile else self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_mpc=None):
		"""
		Select an action by planning in the latent space of the world model,
		or by sampling directly from the policy if MPC is disabled.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
			use_mpc (bool or None): Override cfg.mpc for this call; None uses cfg.mpc.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if use_mpc is None:
			use_mpc = self.cfg.mpc
		if use_mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""
		Estimate the value of a trajectory starting at latent state z and executing
		given actions, using V_pol (rather than Q) as the terminal value.

		Args:
			z (torch.Tensor): Starting latent state.
			actions (torch.Tensor): Sequence of actions to execute (horizon, num_samples, action_dim).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Estimated discounted return for each trajectory.
		"""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		terminal_value = self.model.V(z, task, return_type='avg')
		return G + discount * (1-termination) * terminal_value

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model and MPPI,
		with V_pol (rather than Q) as the terminal value function.

		Args:
			obs (torch.Tensor): Observation from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states. Unlike TD-MPC2's Q-based
		actor loss, the objective here is the expanded model-based gradient through
		reward + dynamics: r(z, pi(z)) + gamma * V_pol(f(z, pi(z))) -- the policy's
		own action is rolled one step through the learned model rather than scored
		against a Q(s,a) critic.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			TensorDict: Loss and diagnostics of the policy update.
		"""
		action, pi_info = self.model.pi(zs, task)
		r_scalar = math.two_hot_inv(self.model.reward(zs, action, task), self.cfg)
		next_z = self.model.next(zs, action, task)
		v_next = self.model.V(next_z, task, return_type='avg', detach=True)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		qs = r_scalar + discount * v_next

		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of expanded-gradient value estimates
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * pi_info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": pi_info["entropy"],
			"pi_scaled_entropy": pi_info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})

	@torch.no_grad()
	def _td_target_2step(self, next_z, reward, terminated, task):
		"""
		2-step TD target: one real buffer step, then one on-policy model expansion.
		  y = r_buffer + gamma*(1-done)*(r_model(next_z, π(next_z)) + gamma*V(f(next_z, π(next_z))))
		First step uses real (s, a, r, s') from the buffer; second step expands forward
		with the learned model under the current policy.
		"""
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		pi_action, _ = self.model.pi(next_z, task)
		next_next_z = self.model.next(next_z, pi_action, task)
		r_pi = math.two_hot_inv(self.model.reward(next_z, pi_action, task), self.cfg)
		bootstrap = self.model.V(next_next_z, task, return_type='min', target=True)
		return reward + discount * (1-terminated) * (r_pi + discount * bootstrap)

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target_2step(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout for consistency loss
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		reward_preds = self.model.reward(_zs, action, task)
		value_preds = self.model.V(_zs, task, return_type='all')
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, vp_unbind) in enumerate(
			zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), value_preds.unbind(1))
		):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for vp_unbind_unbind in vp_unbind.unbind(0):
				value_loss = value_loss + math.soft_ce(vp_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated) if self.cfg.episodic else 0.

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)
		# pi backward flowed through reward/dynamics; clear any stray grads on model params
		self.optim.zero_grad(set_to_none=True)

		# Update target V-function
		self.model.soft_update_target_V()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			TensorDict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		metrics = self._update(obs, action, reward, terminated, **kwargs)
		if isinstance(metrics, TensorDict):
			metrics = metrics.detach().clone().cpu()
		return metrics
