import matplotlib.pyplot as plt
import torch 
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MLP
from torchrl.envs import TransformedEnv, GymEnv, StepCounter, ObservationNorm, DoubleToFloat
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm

torch.manual_seed(0)

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

