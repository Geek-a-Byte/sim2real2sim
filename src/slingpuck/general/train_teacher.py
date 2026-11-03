"""Train the privileged PPO teacher on the goal-conditioned env, with one of the three curricula.

    python -m src.slingpuck.general.train_teacher --curriculum uniform --timesteps 2000000
    python -m src.slingpuck.general.train_teacher --curriculum atr
    python -m src.slingpuck.general.train_teacher --curriculum failure --failures logs/real/trials.jsonl

Writes logs/general/<run_id>/: run_config.yaml, checkpoints, final_model.zip and sim_trials.jsonl (every
training episode as a trial record). The curriculum gets the episode results after each rollout and
gives the envs a new task pool. Use --resume to continue from a model (the next real2sim round).

Warm start from behaviour cloning (general/bc.py):
    python -m src.slingpuck.general.train_teacher --objects puck --init logs/general/bc_<...>/bc_model.zip \
        --critic-warmup 200000 --lr 1e-4
--init copies the BC policy weights into a new PPO model. --critic-warmup first trains only the critic (the
actor is frozen), so that the first PPO updates of the actor use a fitted value function.
"""
import argparse
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback

from src.slingpuck.config import REPO_ROOT, save_run_config
from src.slingpuck.general.common import load_general_config, objects_from_config
from src.slingpuck.general.curriculum import MODES, Curriculum
from src.slingpuck.general.failures import load_jsonl, record_from_info, save_jsonl
from src.slingpuck.general.goal_env import GoalEnv, TaskSpec
from src.slingpuck.general.reachability import ReachabilityChecker, load_or_build_map


class CurriculumCallback(BaseCallback):
    """Feeds finished episodes to the curriculum and gives the envs a new task pool after each rollout."""

    def __init__(self, curriculum: Curriculum, trials_path, pool_size: int):
        super().__init__()
        self.cur, self.trials_path, self.pool_size = curriculum, trials_path, pool_size
        self.results, self.records, self.n_episodes = [], [], 0

    def _on_training_start(self):
        self.training_env.env_method("set_task_pool", self.cur.sample_pool(self.pool_size))

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.n_episodes += 1
                rec = record_from_info(info, f"sim-{self.n_episodes}")
                self.records.append(rec)
                if info.get("start", "normal") == "normal":   # Assisted starts are training aids, not the task
                    self.results.append((TaskSpec.from_dict(info["task"]), rec.success))
        return True

    def _on_rollout_end(self):
        if self.records:
            save_jsonl(self.records, self.trials_path)
            self.records = []
        if self.results:
            n = len(self.results)
            self.logger.record("curriculum/success_rate", sum(s for _, s in self.results) / n)
            self.logger.record("curriculum/episodes", n)
            self.cur.update(self.results)
            self.results = []
        self.training_env.env_method("set_task_pool", self.cur.sample_pool(self.pool_size))


def set_actor_trainable(policy, trainable: bool):
    """Freeze or unfreeze the actor (encoder, MLP, action head, log_std); the critic stays trainable."""
    from src.slingpuck.general.bc import actor_parameters
    for param in [*actor_parameters(policy), policy.log_std]:
        param.requires_grad_(trainable)


def make_env_fn(cfg, rmap, student, objects):
    def fn():
        return GoalEnv(cfg, student=student, reach_map=rmap, objects=objects)
    return fn


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from src.slingpuck.general.policy import PairedPointPolicy

    p = argparse.ArgumentParser()
    p.add_argument("--curriculum", choices=MODES, default="uniform")
    p.add_argument("--failures", help="Trial records (JSONL) with real failures, for --curriculum failure")
    p.add_argument("--objects", nargs="*", help="Object names from configs/general.yaml (default: all)")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int)
    p.add_argument("--resume", help="final_model.zip to continue from")
    p.add_argument("--init", help="bc_model.zip: start the policy from these behaviour-cloning weights")
    p.add_argument("--critic-warmup", type=int, default=0, help="Steps of critic-only training first (with --init)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="Start action noise (std = exp of this; -1 gives 0.37). Run 1 used 0 (std 1): the policy "
                        "saturated its actions to avoid the action-rate penalty")
    args = p.parse_args()

    cfg = load_general_config()
    set_random_seed(args.seed)
    rmap = load_or_build_map(cfg, verbose=True)
    shapes = objects_from_config(cfg)
    failures = load_jsonl(args.failures) if args.failures else []
    if args.curriculum == "failure" and not failures:
        raise SystemExit("--curriculum failure needs --failures with at least one policy failure")
    cur = Curriculum(cfg, ReachabilityChecker(cfg, rmap), shapes, args.curriculum, failures, args.objects, args.seed)

    run_id = f"teacher_{args.curriculum}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = REPO_ROOT / "logs" / "general" / run_id
    save_run_config({**cfg, "train": vars(args)}, run_dir, seed=args.seed)
    env = make_vec_env(make_env_fn(cfg, rmap, False, args.objects), n_envs=args.n_envs, seed=args.seed,
                       vec_env_cls=SubprocVecEnv if args.subproc else DummyVecEnv)
    if args.resume:
        model = PPO.load(args.resume, env=env, custom_objects={"policy_class": PairedPointPolicy})
    else:
        model = PPO(PairedPointPolicy, env, n_steps=512, batch_size=1024, n_epochs=5, gamma=0.99,
                    learning_rate=args.lr, ent_coef=0.0, seed=args.seed, verbose=0,
                    policy_kwargs={"net_arch": [256, 256], "log_std_init": args.log_std_init},
                    tensorboard_log=str(REPO_ROOT / "tensorboard_logs" / "general"))
    callbacks = CallbackList([
        CurriculumCallback(cur, run_dir / "sim_trials.jsonl", args.pool_size or cfg["curriculum"]["pool_size"]),
        CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1), save_path=str(run_dir), name_prefix="model"),
    ])
    if args.init:
        bc = PPO.load(args.init, custom_objects={"policy_class": PairedPointPolicy})
        model.policy.load_state_dict(bc.policy.state_dict())
        print(f"Policy initialised from {args.init}")
    print(f"Training {run_id}")
    warmup = min(args.critic_warmup, args.timesteps) if args.init else 0
    if warmup:
        set_actor_trainable(model.policy, False)
        model.learn(total_timesteps=warmup, callback=callbacks, tb_log_name=run_id, reset_num_timesteps=True)
        set_actor_trainable(model.policy, True)
        print(f"Critic warm-up done ({warmup} steps); the actor now trains too")
    model.learn(total_timesteps=args.timesteps - warmup, callback=callbacks, tb_log_name=run_id,
                reset_num_timesteps=not (args.resume or warmup))
    model.save(run_dir / "final_model")
    env.close()
    print(f"Saved {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
