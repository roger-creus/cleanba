import functools
import os
import queue
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import chex
import envpool
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from rich.pretty import pprint
from tensorboardX import SummaryWriter

from cleanrl_utils.atari_scores import ATARI_SCORES

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
)
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
)
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    "the name of this experiment"
    seed: int = 1
    "seed of the experiment"
    track: bool = False
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "cleanba"
    "the wandb's project name"
    wandb_entity: str = None
    "the entity (team) of wandb's project"
    capture_video: bool = False
    "whether to capture videos of the agent performances (check out `videos` folder)"
    save_model: bool = False
    "whether to save model into the `runs/{run_name}` folder"
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    log_frequency: int = 10
    "the logging frequency of the model performance (in terms of `updates`)"

    # Algorithm specific hparams
    env_id: str = "Breakout-v5"
    "the id of the environment"
    total_timesteps: int = 50000000
    "total timesteps of the experiments"
    total_decay_timesteps: int = 50000000
    "total timesteps of the experiments"
    local_num_envs: int = 128
    "the number of parallel game environments"
    num_actor_threads: int = 2
    "the number of actor threads to use"
    num_steps: int = 32
    "the number of steps to run in each environment per policy rollout"
    lr: float = 0.00025
    "initial learning rate"
    anneal_lr: bool = False
    "Toggle learning rate annealing for Q network"
    eps_start: float = 1.0
    "epsilon start"
    eps_end: float = 0.001
    "epsilon end"
    eps_decay: float = 0.1
    "epsilon decay"
    gamma: float = 0.99
    "the discount factor gamma"
    num_minibatches: int = 32
    "the number of mini-batches"
    gradient_accumulation_steps: int = 1
    "the number of gradient accumulation steps before performing an optimization step"
    update_epochs: int = 2
    "the K epochs to update the policy"
    norm_type: str = "layer_norm"
    "normalization type in Q network"
    norm_input: bool = False
    "whether to normalize input before any forward pass"
    max_grad_norm: float = 10.0
    "the maximum norm for the gradient clipping"
    lmbda: float = 0.65
    "lambda parameter for Q(lambda) targets."

    # Distributed args
    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that learner workers will use"
    distributed: bool = False
    "whether to use `jax.distirbuted`"
    concurrency: bool = False
    "whether to run the actor and learner concurrently"

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0
    global_learner_decices: Optional[List[str]] = None
    actor_devices: Optional[List[str]] = None
    learner_devices: Optional[List[str]] = None


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping


def make_env(env_id: str, seed: int, num_envs: int) -> Callable[..., Any]:
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,  # lost life -- done, increases sample efficiency, may hurt in some games
            repeat_action_probability=0.0,  # stay deterministic, as it is Q learning
            noop_max=30,
            reward_clip=True,
            seed=seed,
            frame_skip=4,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        envs.name = env_id

        return envs

    return thunk


# ======== Neural networks ========


class CNN(nn.Module):
    """Conv torso for DQN, with additional norms."""

    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    """Full DQN."""

    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


# ========= Algorithm stuff =========


class Transition(NamedTuple):
    obs: list
    next_obs: list
    dones: list
    actions: list
    env_ids: list
    rewards: list
    q_values: list
    truncations: list
    terminations: list
    firststeps: list  # first step of an episode


class CustomTrainState(TrainState):

    batch_stats: Any
    grad_steps: int = 0


def eps_greedy_exploration(
    rng: jax.random.PRNGKey, q_values: jax.Array, epsilon: float
) -> jax.Array:
    """Epsilon-greedy exploration."""

    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_values, axis=-1)
    chosen_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < epsilon,
        jax.random.randint(
            rng_a, greedy_actions.shape, minval=0, maxval=q_values.shape[-1]
        ),
        greedy_actions,
    )
    return chosen_actions


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    eps_schedule_fn: optax.Schedule,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    batch_stats_queue: queue.Queue,
    writer: SummaryWriter,
    learner_devices: List,
    device_thread_id: int,
    actor_device: int,
):

    envs = make_env(
        args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
    )()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    @jax.jit
    def get_action(
        params: flax.core.FrozenDict,
        batch_stats: flax.core.FrozenDict,
        obs: np.ndarray,
        eps: float,
        key: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.random.PRNGKey]:
        """Action function."""

        obs = jnp.array(obs)
        chex.assert_rank(obs, 4)  # [num_envs, 4, 84, 84]

        q_values = QNetwork(
            action_dim=envs.single_action_space.n,
            norm_type=args.norm_type,
            norm_input=args.norm_input,
        ).apply({"params": params, "batch_stats": batch_stats}, obs, train=False)

        # do eps greedy exploration
        key, subkey = jax.random.split(key)
        actions = eps_greedy_exploration(subkey, q_values, eps)

        # now return obs, actions, q values, key
        return obs, actions, q_values, key

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    obs = envs.reset()
    done = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree_map(
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if args.concurrency:
            if update != 2:
                params = params_queue.get()
                # NOTE: block here is important because otherwise this thread will call
                # the jitted `get_action_and_value` function that hangs until the params are ready.
                # This blocks the `get_action_and_value` function in other actor threads.
                # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                params.network_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                actor_policy_version += 1

                # also grab batch stats from queue
                batch_stats = batch_stats_queue.get()
        else:
            params = params_queue.get()
            batch_stats = batch_stats_queue.get()
            actor_policy_version += 1

        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        storage = []
        for _ in range(0, args.num_steps):
            cached_obs = obs
            cached_done = done
            global_step += (
                len(done)
                * args.num_actor_threads
                * len_actor_device_ids
                * args.world_size
            )
            eps = eps_schedule_fn(global_step)

            inference_time_start = time.time()
            cached_obs, action, q_value, key = get_action(
                params, batch_stats, cached_obs, eps, key
            )
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            cpu_action = np.array(action)
            d2h_time += time.time() - d2h_time_start

            env_send_time_start = time.time()
            next_obs, next_reward, done, info = envs.step(cpu_action)
            env_id = info["env_id"]
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            # set obs to next obs -- completes loop and makes sure `cached_obs` is right next time
            obs = next_obs

            # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
            # so we use our own truncated flag
            truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
            storage.append(
                Transition(
                    obs=cached_obs,
                    next_obs=next_obs,
                    dones=cached_done,
                    actions=action,
                    env_ids=env_id,
                    rewards=next_reward,
                    q_values=q_value,
                    truncations=truncated,
                    terminations=info["terminated"],
                    firststeps=info["elapsed_step"] == 0,
                )
            )

            episode_returns[env_id] += info["reward"]
            returned_episode_returns[env_id] = np.where(
                info["terminated"] + truncated,
                episode_returns[env_id],
                returned_episode_returns[env_id],
            )
            episode_returns[env_id] *= (1 - info["terminated"]) * (1 - truncated)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                info["terminated"] + truncated,
                episode_lengths[env_id],
                returned_episode_lengths[env_id],
            )
            episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - truncated)
            storage_time += time.time() - storage_time_start
        rollout_time.append(time.time() - rollout_time_start)

        avg_episodic_return = np.mean(returned_episode_returns)
        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *list(
                map(
                    lambda x: jax.device_put_sharded(x, devices=learner_devices),
                    partitioned_storage,
                )
            )
        )

        # obs, done are still in the host (these are last values)
        sharded_obs = jax.device_put_sharded(
            np.split(obs, len(learner_devices)), devices=learner_devices
        )
        sharded_done = jax.device_put_sharded(
            np.split(done, len(learner_devices)), devices=learner_devices
        )
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            sharded_obs,
            sharded_done,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        if update % args.log_frequency == 0:
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar(
                "charts/avg_episodic_return", avg_episodic_return, global_step
            )
            writer.add_scalar(
                "charts/avg_episodic_length",
                np.mean(returned_episode_lengths),
                global_step,
            )
            writer.add_scalar(
                "stats/params_queue_get_time",
                np.mean(params_queue_get_time),
                global_step,
            )
            writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/storage_time", storage_time, global_step)
            writer.add_scalar("stats/d2h_time", d2h_time, global_step)
            writer.add_scalar("stats/env_send_time", env_send_time, global_step)
            writer.add_scalar(
                "stats/rollout_queue_put_time",
                np.mean(rollout_queue_put_time),
                global_step,
            )
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            writer.add_scalar(
                "charts/SPS_update",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * args.num_actor_threads
                    * args.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.local_batch_size = int(
        args.local_num_envs
        * args.num_steps
        * args.num_actor_threads
        * len(args.actor_device_ids)
    )
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids))
        * args.num_actor_threads
        % args.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(
                len(args.learner_device_ids) + len(args.actor_device_ids)
            ),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = (
        args.local_num_envs
        * args.world_size
        * args.num_actor_threads
        * len(args.actor_device_ids)
    )
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    if args.track and args.local_rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key = jax.random.split(key)
    learner_keys = jax.device_put_replicated(key, learner_devices)

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs)()

    # schedules
    num_decay_updates = args.total_decay_timesteps // args.num_steps // args.num_envs
    eps_scheduler = optax.linear_schedule(
        args.eps_start, args.eps_end, args.eps_decay * num_decay_updates
    )
    lr_scheduler = optax.linear_schedule(
        init_value=args.lr,
        end_value=1e-20,
        transition_steps=num_decay_updates * args.num_minibatches * args.update_epochs,
    )
    lr = lr_scheduler if args.anneal_lr else args.lr

    # network
    network = QNetwork(
        action_dim=envs.single_action_space.n,
        norm_type=args.norm_type,
        norm_input=args.norm_input,
    )
    network_variables = network.init(
        network_key, np.array([envs.single_observation_space.sample()]), train=False
    )
    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.radam)(learning_rate=lr),
        ),
        every_k_schedule=args.gradient_accumulation_steps,
    )
    agent_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )
    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)

    # log
    print(
        network.tabulate(
            network_key, np.array([envs.single_observation_space.sample()]), train=False
        )
    )

    # define the losses
    @jax.jit
    def _compute_targets(
        last_q: jax.Array, q_vals: jax.Array, reward: jax.Array, done: jax.Array
    ) -> jax.Array:
        """Q(lambda) target values."""

        # with jax.disable_jit():
        #     print(last_q.shape)
        #     print(q_vals.shape)
        #     print(reward.shape)
        #     print(done.shape)

        def _get_target(
            lam_returns_and_next_q: Tuple[jax.Array, jax.Array],
            rew_q_done: Tuple[jax.Array, jax.Array, jax.Array],
        ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
            reward, q, done = rew_q_done
            lambda_returns, next_q = lam_returns_and_next_q

            target_bootstrap = reward + args.gamma * (1 - done) * next_q
            delta = lambda_returns - next_q
            lambda_returns = target_bootstrap + args.gamma * args.lmbda * delta
            lambda_returns = (1 - done) * lambda_returns + done * reward
            next_q = jnp.max(q, axis=-1)
            return (lambda_returns, next_q), lambda_returns

        lambda_returns = reward[-1] + args.gamma * (1 - done[-1]) * last_q
        last_q = jnp.max(q_vals[-1], axis=-1)
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, last_q),
            jax.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
            reverse=True,
        )
        targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
        return targets

    def pqn_loss(
        params: flax.core.FrozenDict,
        batch_stats: flax.core.FrozenDict,
        obs: jax.Array,
        actions: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        """PQN loss."""

        q_values, updates = QNetwork(
            action_dim=envs.single_action_space.n,
            norm_type=args.norm_type,
            norm_input=args.norm_input,
        ).apply(
            {"params": params, "batch_stats": batch_stats},
            obs,
            train=True,
            mutable=["batch_stats"],
        )

        # get regression predictions and targets
        chosen_action_qvals = jnp.take_along_axis(
            q_values, jnp.expand_dims(actions, axis=-1), axis=-1
        ).squeeze(axis=-1)

        # compute loss and return
        loss = 0.5 * jnp.square(chosen_action_qvals - targets).mean()
        return loss, (updates, chosen_action_qvals)

    @jax.jit
    def single_device_update(
        agent_state: CustomTrainState,
        sharded_storages: List,
        sharded_obs: List,
        sharded_done: List,
        key: jax.random.PRNGKey,
    ):
        """Single-device update."""

        # not needed as all values for bootstrapping are in storage
        del sharded_obs, sharded_done

        storage: Transition = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        # obs = jnp.concatenate(sharded_obs)
        # done = jnp.concatenate(sharded_done)

        pqn_loss_grad_fn = jax.value_and_grad(pqn_loss, has_aux=True)

        # now grab targets for the loss
        last_q = network.apply(
            {"params": agent_state.params, "batch_stats": agent_state.batch_stats},
            storage.next_obs[-1],
            train=False,
        )
        last_q = jnp.max(last_q, axis=-1)
        targets = _compute_targets(
            last_q, storage.q_values, storage.rewards, storage.dones
        )

        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(
                    x,
                    (args.num_minibatches * args.gradient_accumulation_steps, -1)
                    + x.shape[1:],
                )
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (mb_obs, mb_actions, mb_targets) = minibatch

                (loss, (updates, chosen_action_qvals)), grads = pqn_loss_grad_fn(
                    agent_state.params,
                    agent_state.batch_stats,
                    mb_obs,
                    mb_actions,
                    mb_targets,
                )
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)

                # also replace batch stats and grad steps
                agent_state = agent_state.replace(
                    grad_steps=agent_state.grad_steps + 1,
                    batch_stats=updates["batch_stats"],
                )

                return agent_state, (loss, chosen_action_qvals)

            agent_state, (loss, chosen_action_qvals) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (shuffled_storage.obs, shuffled_storage.actions, targets),
            )
            return (agent_state, key), (loss, chosen_action_qvals)

        (agent_state, key), (loss, chosen_action_qvals) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        chosen_action_qvals = jax.lax.pmean(
            chosen_action_qvals, axis_name="local_devices"
        ).mean()
        return agent_state, loss, chosen_action_qvals, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    )

    params_queues = []
    rollout_queues = []
    batch_stats_queues = []
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    unreplicated_batch_stats = flax.jax_utils.unreplicate(agent_state.batch_stats)
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        device_batch_stats = jax.device_put(
            unreplicated_batch_stats, local_devices[d_id]
        )
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            batch_stats_queues.append(queue.Queue(maxsize=1))

            params_queues[-1].put(device_params)
            batch_stats_queues[-1].put(device_batch_stats)

            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    eps_scheduler,
                    rollout_queues[-1],
                    params_queues[-1],
                    batch_stats_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    d_idx * args.num_actor_threads + thread_id,
                    local_devices[d_id],
                ),
            ).start()

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_dones = []
        for d_idx, d_id in enumerate(args.actor_device_ids):
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    avg_params_queue_get_time,
                    device_thread_id,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_dones.append(sharded_next_done)
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (agent_state, loss, chosen_action_qvals, learner_keys) = multi_device_update(
            agent_state,
            sharded_storages,
            sharded_next_obss,
            sharded_next_dones,
            learner_keys,
        )

        # put stuff back into queue
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        unreplicated_batch_stats = flax.jax_utils.unreplicate(agent_state.batch_stats)

        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            device_batch_stats = jax.device_put(
                unreplicated_batch_stats, local_devices[d_id]
            )
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(
                    device_params
                )
                batch_stats_queues[d_idx * args.num_actor_threads + thread_id].put(
                    device_batch_stats
                )

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar(
                "stats/rollout_queue_get_time",
                np.mean(rollout_queue_get_time),
                global_step,
            )
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar(
                "stats/training_time", time.time() - training_time_start, global_step
            )
            writer.add_scalar(
                "stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step
            )
            writer.add_scalar(
                "stats/params_queue_size", params_queues[-1].qsize(), global_step
            )
            print(
                global_step,
                f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
            )
            writer.add_scalar(
                "charts/learning_rate",
                agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(),
                global_step,
            )
            writer.add_scalar("losses/pqn_loss", loss[-1].item(), global_step)
        if learner_policy_version >= args.num_updates:
            break

    if args.save_model and args.local_rank == 0:
        if args.distributed:
            jax.distributed.shutdown()
        agent_state = flax.jax_utils.unreplicate(agent_state)
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [agent_state.params],
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(QNetwork,),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
                extra_dependencies=["jax", "envpool", "atari"],
            )

    envs.close()
    writer.close()
