import matplotlib.pyplot as plt
import torch 
from torch.optim import Adam
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, AdditiveGaussianModule
from torchrl.envs import TransformedEnv, GymEnv, StepCounter, ObservationNorm, DoubleToFloat, Compose
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler
from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm

torch.manual_seed(0)

"""
done:
    - wrote environment
    - wrote policy network
    - write critic network (Q-function Network (target and normal))
        - Critic network takes state,action combined
    - Data collector
    - Replay Buffer
todo:
    - Training
"""



def make_env():
    base_env = TransformedEnv(GymEnv('BipedalWalker-v3'))
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter()
        )
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    return env

env = make_env()

device = torch.device("cpu")

frames_per_batch = 1000
total_frames = 10_000 #testing purposes, this is nowhere near enough
sub_batch_size = 64

optim_steps = 10 #number of times to update the critic and actor networks

memory_size = 1_000_000 #replay buffer size (DDPG is off-policy)

num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

train_batch_size = 128  # Number of frames trained in each optimiser step

gamma = 0.99 #discount factor
polyak = 0.005 #soft update rate for target network


policy = MLP(
    in_features=env.observation_spec["observation"].shape[-1],
    out_features=env.action_spec.shape[-1],
    depth=3,
    num_cells=num_cells,
    activation_class=torch.nn.Tanh
)

policy_module = TensorDictModule(
    policy,
    in_keys=["observation"],
    out_keys=["action"]
)


#exploration_module for training since ddpg is deterministic
exploration_module = TensorDictSequential(
    policy_module,
    AdditiveGaussianModule(
        spec=env.action_spec,
        annealing_num_steps=total_frames
    )
)

cat_module = TensorDictModule(
    lambda obs,act : torch.cat([obs,act],dim=-1),
    in_keys=["observation","action"],
    out_keys=["obs_act"]
)

critic_module = TensorDictModule(
    MLP(
        in_features=env.observation_spec["observation"].shape[-1] + env.action_spec.shape[-1],
        out_features=1,
        depth=3,
        num_cells=num_cells,
        activation_class=torch.nn.Tanh
    ),
    in_keys=["obs_act"],
    out_keys=["state_action_value"]
)

#critic network 
critic = TensorDictSequential(
    cat_module,
    critic_module
)

#im not sure if this is neede
target_critic = TensorDictSequential(
    cat_module,
    critic_module
)

collector = SyncDataCollector(
    env,
    exploration_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device
)

#replay buffer for off-policy training
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=memory_size,),
    sampler=RandomSampler(),
) 

loss = DDPGLoss(
    actor_network=policy_module,
    value_network=critic,
)

# optim = Adam(loss.parameters(), lr=lr)

actor_optimizer = Adam(loss.actor_network_params.flatten_keys().values(), lr=lr)
value_optimizer = Adam(loss.value_network_params.flatten_keys().values(), lr=lr)

updater = SoftUpdate(loss,tau=polyak)


#havent updated both networks ???? why not 
for i, tensordict_data in enumerate(collector):
    #add to replay buffer
    current_frame = tensordict_data.numel()
    replay_buffer.extend(tensordict_data)
    # batch = replay_buffer.sample(batch_size=frames_per_batch)
    for _ in range(optim_steps):
        #sample from replay buffer
        batch = replay_buffer.sample(batch_size=train_batch_size)
        #compute the loss
        loss_vals = loss(batch)

        actor_loss = loss_vals["loss_actor"]
        actor_loss.backward()
        params_a = actor_optimizer.param_groups[0]["params"]
        torch.nn.utils.clip_grad_norm_(params_a, max_grad_norm)
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        value_loss = loss_vals["loss_value"]
        value_loss.backward()
        params_l = value_optimizer.param_groups[0]["params"]
        torch.nn.utils.clip_grad_norm_(params_l, max_grad_norm)
        value_optimizer.step()
        value_optimizer.zero_grad()

        updater.step()

    exploration_module[-1].step(current_frame)

        # for loss_name in ["loss_actor", "loss_value"]:
            # loss_vals[loss_name].backward()
            # loss_i = loss_vals[loss_name]
            # optim


    # loss_vals = loss(batch)
    # loss_vals["loss"].backward()
    # optim.step()
    # optim.zero_grad()
    # updater.step()
    
    if i % 100 == 0:
        print(f"Iteration {i}: Actor Loss: {loss_vals}")
        print(f"Iteration {i}: batch: {batch}")
        print(f"Iteration {i}: Action: {batch['action'].mean().item()}")
    