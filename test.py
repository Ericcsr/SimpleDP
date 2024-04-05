import os
import json
import time
import numpy as np
import torch
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from neurals import ConditionalUnet1D
from datasets import normalize_data, unnormalize_data, PushTStateDataset
from envs import PushTEnv
from tqdm.auto import tqdm

from skvideo.io import vwrite
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--ckpt", type=str, default="latest")
parser.add_argument("--max_steps", type=int, default=200)
args = parser.parse_args()

config = json.load(open(f"configs/{args.config}.json"))
obs_dim = config['obs_dim']
action_dim = config['action_dim']
pred_horizon = config['pred_horizon']
obs_horizon = config['obs_horizon']
action_horizon = config['action_horizon']
num_diffusion_iters = config['num_diffusion_iters']
dataset_path = config['dataset_path']

device = torch.device('cuda')

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
).to(device)

# load dataset
# create dataset from file
dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)



ckpt_path = f"checkpoints/{args.ckpt}.pth"
if not os.path.isfile(ckpt_path):
    print("Please first train the model to get the checkpoint.")
    exit(-1)

state_dict = torch.load(ckpt_path, map_location='cuda')
ema_noise_pred_net = noise_pred_net
ema_noise_pred_net.load_state_dict(state_dict)
print('Pretrained weights loaded.')


#@markdown ### **Inference**

# limit enviornment interaction to 200 steps before termination
env = PushTEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100000)

# get first observation
obs, info = env.reset()

# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

with tqdm(total=args.max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > args.max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print('Score: ', max(rewards))

# save video.
output_dir = f"outputs/{time.time()}"
os.makedirs(output_dir)
vwrite(f"{output_dir}/video.mp4", imgs)