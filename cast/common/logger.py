import dataclasses
import os
import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from termcolor import colored

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def write_seed_results(log_dir, task, exp):
	"""
	Aggregate the per-seed eval.csv files for (task, exp) into a baseline-style
	results/<exp>/<task>.csv with columns step,reward,seed.
	Rebuilds from whatever seeds currently exist on disk; written atomically so concurrent
	seeds finishing at once can't corrupt it.
	"""
	log_dir = Path(log_dir)
	task_dir = log_dir.parent.parent          # .../logs/<task>
	repo = log_dir.parents[3]                 # repo root (parent of 'logs')
	rows = []
	for ev in sorted(task_dir.glob(f"*/{exp}/eval.csv")):
		seed = ev.parent.parent.name          # the <seed> directory
		try:
			d = pd.read_csv(ev)
		except Exception:
			continue
		if "step" not in d.columns or len(d.columns) < 2:
			continue
		col = "episode_reward" if "episode_reward" in d.columns else d.columns[1]
		d = d[["step", col]].rename(columns={col: "reward"})
		d["step"] = d["step"].round().astype("int64")
		d["seed"] = seed
		rows.append(d[["step", "reward", "seed"]])
	if not rows:
		return None
	out_df = pd.concat(rows, ignore_index=True).sort_values(["seed", "step"])
	out_dir = repo / "results" / exp
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{task}.csv"
	tmp = out_path.with_suffix(".csv.tmp")
	out_df.to_csv(tmp, index=False)
	os.replace(tmp, out_path)
	return out_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=50):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._task = cfg.task
		self._exp = cfg.exp_name
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		# Written directly to disk at the very start of training, independent of
		# wandb's own config.yaml (which wandb only writes locally on a clean
		# run.finish()/exit — never on kill, and per direct testing, never
		# incrementally during a run either). eval_r1.py reads this first, before
		# falling back to wandb's copy.
		OmegaConf.save(config=OmegaConf.create(dataclasses.asdict(cfg)), f=str(self._log_dir / "config.yaml"))
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			name=str(cfg.seed),
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._save_csv:
			try:
				out = write_seed_results(self._log_dir, self._task, self._exp)
				if out is not None:
					print(colored(f"Aggregated seed results -> {out}", "blue"))
			except Exception as e:
				print(colored(f"Failed to write aggregated results CSV: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train", print_console=True):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			xkey = "step"
			_d = {category + "/" + k: v for k, v in d.items()}
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			if "policy_episode_reward" in d:
				keys.append("policy_episode_reward")
			self._eval.append(np.array([d[k] for k in keys]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		if print_console:
			self._print(d, category)

	def log_reward_terms(self, d, split, step):
		"""Logs per-reward-term breakdowns (env info keys prefixed 'reward/') under
		their own top-level wandb group ('reward_terms/<split>/...') instead of
		nesting them inside the main train/eval groups, so they show up as a
		separately collapsible section in the wandb UI."""
		if self._wandb and d:
			_d = {f"reward_terms/{split}/{k[len('reward/'):]}": v for k, v in d.items()}
			self._wandb.log(_d, step=step)
