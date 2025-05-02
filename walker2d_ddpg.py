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
            ObservationNorm(),
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
num_epochs = 10
memory_size = 1_000_000 #replay buffer size (DDPG is off-policy)

num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

train_batch_size = 128  # Number of frames trained in each optimiser step

gamma = 0.99 #discount factor
polyak = 0.005 #soft update rate for target network


policy = MLP(
    in_features=env.observation_spec,
    out_features=env.action_spec,
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
        in_features=env.observation_spec + env.action_spec,
        out_features=1,
        depth=3,
        num_cells=num_cells,
        activation_class=torch.nn.Tanh
    ),
    in_keys=["obs_act"],
    out_keys=["q_value"]
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
    device=device
) 

loss = DDPGLoss(
    actor_network=policy_module,
    value_network=critic_module,
    device=device
)

optim = Adam(loss.parameters(), lr=lr)
updater = SoftUpdate(loss,eps=0.99,tau=polyak)

