from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from tqdm import tqdm
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2-Vpi training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._elapsed_before_resume = 0.0
		self._checkpoint_dir = Path(self.cfg.work_dir) / "checkpoint"

	def common_metrics(self):
		elapsed = self._elapsed_before_resume + (time() - self._start_time)
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed,
			steps_per_second=self._step / max(elapsed, 1e-8),
		)

	def save_checkpoint(self):
		"""Bundle everything needed for a true resume (model, both optimizers, buffer
		contents, trainer step/episode counters) -- unlike Logger.save_agent(), which
		only ever persists model weights for periodic/final export. Metadata (task,
		seed, exp_name, work_dir) is stored alongside so a checkpoint can be traced back
		to the run that produced it even if copied elsewhere."""
		self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.buffer.save_checkpoint(str(self._checkpoint_dir / "buffer"))
		torch.save({
			"model": self.agent.model.state_dict(),
			"optim": self.agent.optim.state_dict(),
			"pi_optim": self.agent.pi_optim.state_dict(),
			"step": self._step,
			"ep_idx": self._ep_idx,
			"elapsed_time": self._elapsed_before_resume + (time() - self._start_time),
			"buffer_num_eps": self.buffer.num_eps,
			"task": self.cfg.task,
			"seed": self.cfg.seed,
			"exp_name": self.cfg.exp_name,
			"work_dir": str(self.cfg.work_dir),
		}, self._checkpoint_dir / "state.pt")

	def load_checkpoint(self, checkpoint_dir):
		"""Restore a checkpoint written by save_checkpoint() from checkpoint_dir
		(explicit path, e.g. logs/<task>/<seed>/<exp_name>/checkpoint). Warns (does not
		refuse) if the checkpoint's recorded task/seed don't match the current run,
		since intentionally warm-starting from a different run's checkpoint is a valid
		use case. Returns True if loaded, False if checkpoint_dir has no checkpoint."""
		checkpoint_dir = Path(checkpoint_dir)
		state_fp = checkpoint_dir / "state.pt"
		if not state_fp.exists():
			raise FileNotFoundError(f"No checkpoint found at {state_fp}")
		state = torch.load(state_fp, map_location=self.agent.device)
		if state.get("task") != self.cfg.task or state.get("seed") != self.cfg.seed:
			print(f"Warning: resuming from a checkpoint recorded as "
				  f"task={state.get('task')} seed={state.get('seed')} exp_name={state.get('exp_name')}, "
				  f"but this run is task={self.cfg.task} seed={self.cfg.seed} exp_name={self.cfg.exp_name}.")
		self.agent.model.load_state_dict(state["model"])
		self.agent.optim.load_state_dict(state["optim"])
		self.agent.pi_optim.load_state_dict(state["pi_optim"])
		self._step = state["step"]
		self._ep_idx = state["ep_idx"]
		self._elapsed_before_resume = state["elapsed_time"]
		self.buffer.load_checkpoint(str(checkpoint_dir / "buffer"), state["buffer_num_eps"],
									 device=self.agent.device.type)
		return True

	def eval(self):
		def run_eval(use_mpc, save_video=False):
			ep_rewards, ep_successes, ep_lengths = [], [], []
			reward_term_sums = {}
			for i in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(), False, 0, 0
				if save_video and self.cfg.save_video:
					self.logger.video.init(self.env, enabled=(i==0))
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, use_mpc=use_mpc)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
					for k, v in info.items():
						if k.startswith('reward/'):
							reward_term_sums[k] = reward_term_sums.get(k, 0.0) + v
					if save_video and self.cfg.save_video:
						self.logger.video.record(self.env)
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
				ep_lengths.append(t)
				if save_video and self.cfg.save_video:
					self.logger.video.save(self._step)
			out = dict(
				reward=np.nanmean(ep_rewards),
				success=np.nanmean(ep_successes),
				length=np.nanmean(ep_lengths),
			)
			total_steps = sum(ep_lengths)
			for k, v in reward_term_sums.items():
				out[k] = v / max(total_steps, 1)
			return out

		mpc_metrics = run_eval(use_mpc=True, save_video=True)
		policy_metrics = run_eval(use_mpc=False, save_video=False)
		metrics = dict(
			episode_reward=mpc_metrics['reward'],
			episode_success=mpc_metrics['success'],
			episode_length=mpc_metrics['length'],
			policy_episode_reward=policy_metrics['reward'],
			policy_episode_success=policy_metrics['success'],
			policy_episode_length=policy_metrics['length'],
		)
		metrics['_reward_terms'] = {k: v for k, v in mpc_metrics.items() if k.startswith('reward/')}
		return metrics

	def to_td(self, obs, action=None, reward=None, terminated=None):
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		train_metrics, done, eval_next = {}, True, False
		have_episode_in_progress = False

		resume_path = self.cfg.get('resume', False)
		if resume_path:
			self.load_checkpoint(resume_path)
			print(f"Resumed from checkpoint {resume_path} at step {self._step:,} (episode {self._ep_idx:,})")

		pbar = tqdm(total=self.cfg.steps, initial=self._step, unit="step", dynamic_ncols=True)
		last_ep_time = time()

		while self._step <= self.cfg.steps:
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			if done:
				if eval_next:
					eval_metrics = self.eval()
					reward_terms = eval_metrics.pop('_reward_terms')
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval', print_console=False)
					self.logger.log_reward_terms(reward_terms, 'eval', step=self._step)
					self.logger.save_agent(self.agent, identifier=self._step)
					eval_next = False
					pbar.set_postfix(eval_R=f"{eval_metrics['episode_reward']:.1f}")
					if self.cfg.get('save_checkpoint', False) and self._step > 0:
						self.save_checkpoint()

				if self._step > 0 and have_episode_in_progress:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but episodic=false. Set episodic=true to enable.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					n_ep_steps = len(self._tds) - 1
					reward_terms = {k: v / max(n_ep_steps, 1) for k, v in self._reward_term_sums.items()}
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train', print_console=False)
					self.logger.log_reward_terms(reward_terms, 'train', step=self._step)
					self._ep_idx = self.buffer.add(torch.cat(self._tds))
					now = time()
					ep_time = now - last_ep_time
					last_ep_time = now
					pbar.set_postfix(train_R=f"{train_metrics['episode_reward']:.1f}", ep=self._ep_idx,
									  s_per_ep=f"{ep_time:.1f}")

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]
				self._reward_term_sums = {}

			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			have_episode_in_progress = True
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))
			for k, v in info.items():
				if k.startswith('reward/'):
					self._reward_term_sums[k] = self._reward_term_sums.get(k, 0.0) + v

			if self._step >= self.cfg.seed_steps:
				num_updates = self.cfg.seed_steps if self._step == self.cfg.seed_steps else 1
				if self._step == self.cfg.seed_steps:
					print('Pretraining agent on seed data...')
				if self.buffer.num_eps > 0:
					for _ in range(num_updates):
						_train_metrics = self.agent.update(self.buffer)
					train_metrics.update(_train_metrics)

			self._step += 1
			pbar.update(1)

		pbar.close()
		self.logger.finish(self.agent)
