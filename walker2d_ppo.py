import multiprocessing

def main():
    # All your current code goes here:
    # from imports ...
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import torchrl
    import time
    from torchrl.envs import TransformedEnv, GymEnv, StepCounter, Compose, ObservationNorm, DoubleToFloat, ParallelEnv
    from tensordict.nn import TensorDictModule, TensorDictSequential
    from tensordict.nn.distributions import NormalParamExtractor
    from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.objectives.value import GAE
    from torch.optim import Adam
    from torchrl.objectives import ClipPPOLoss
    from torchrl._utils import logger as torchrl_logger
    from torchrl.record import CSVLogger, VideoRecorder
    from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
    from tqdm import tqdm

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    num_cells = 256
    lr = 3e-4
    max_grad_norm = 1.0

    torch.manual_seed(0)

    def make_env():
        base_env = TransformedEnv(GymEnv('BipedalWalker-v3', device=device))
        env = TransformedEnv(
            base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )
        # print(env.transform)
        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        return env

    num_envs = 6
    env = ParallelEnv(num_envs, make_env)

    frames_per_batch = 1000
    total_frames = 10_000_000
    sub_batch_size = 64
    num_epochs = 10
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor()
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec_unbatched.space.low,
            "high": env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(module=value_net, in_keys=["observation"])

    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        # pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()

    path = "./ppo"
    logger = CSVLogger(exp_name="ppo", log_dir=path, video_format="mp4")
    video_recorder = VideoRecorder(logger, tag="video")

    base_env2 = TransformedEnv(GymEnv('BipedalWalker-v3', device=device, from_pixels=True, pixels_only=False), video_recorder)
    env2 = TransformedEnv(
        base_env2,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env2.transform[1].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    env2.rollout(max_steps=1_000_000, policy=policy_module)
    video_recorder.dump()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # safer on Windows/CUDA
    main()
