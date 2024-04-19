import os
import json
import time
import numpy as np
import torch
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from neurals import ConditionalUnet1D, create_vision_encoder
from datasets import normalize_data, unnormalize_data, PushTStateDataset, PushTImageDataset
from envs import PushTEnv, PushTImageEnv
from tqdm.auto import tqdm

from skvideo.io import vwrite
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="push_t_state")
parser.add_argument("--use_goal", action='store_true', default=False)
parser.add_argument("--ckpt", type=str, default="latest")
parser.add_argument("--max_steps", type=int, default=200)
args = parser.parse_args()

config = json.load(open(f"configs/{args.config}.json"))
if config['type'] == 'state':
    obs_dim = config['obs_dim']
else:
    lowdim_obs_dim = config['lowdim_obs_dim']
    vision_feature_dim = config['vision_feature_dim']
    obs_dim = lowdim_obs_dim + vision_feature_dim
action_dim = config['action_dim']
pred_horizon = config['pred_horizon']
obs_horizon = config['obs_horizon']
action_horizon = config['action_horizon']
num_diffusion_iters = config['num_diffusion_iters_train']
num_diffusion_iters_test = config['num_diffusion_iters_test']
dataset_path = config['dataset_path']

device = torch.device('cuda')

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon + (config['goal_dim'] if args.use_goal else 0),
).to(device)

if config['type'] == 'image':
    vision_encoder = create_vision_encoder().to(device)
    nets = torch.nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
else:
    nets = torch.nn.ModuleDict({
        'noise_pred_net': noise_pred_net
    })

# load dataset
# create dataset from file
if config['type'] == 'state':
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_goal=args.use_goal
    )
else:
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_goal=args.use_goal)
# save training data statistics (min, max) for each dim
stats = dataset.stats

if config['scheduler'] == 'ddpm':
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
elif config["scheduler"] == 'ddim':
    noise_scheduler = DDIMScheduler( # TODO: Change default settings.
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )


if args.use_goal:
    ckpt_path = f"checkpoints/{args.ckpt}_{config['type']}_goal.pth"
else:
    ckpt_path = f"checkpoints/{args.ckpt}_{config['type']}.pth"
if not os.path.isfile(ckpt_path):
    print("Please first train the model to get the checkpoint.")
    exit(-1)

state_dict = torch.load(ckpt_path, map_location='cuda')
ema_nets = nets
ema_nets.load_state_dict(state_dict)

print('Pretrained weights loaded.')


#@markdown ### **Inference**

# limit enviornment interaction to 200 steps before termination
if config['type'] == 'state':
    env = PushTEnv()
else:
    env = PushTImageEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(5000)

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

env_name = 'PushTStateEnv' if config['type'] == 'state' else 'PushTImageEnv'
with tqdm(total=args.max_steps, desc=f"Eval {env_name}") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        if config['type'] == 'state':
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nstate_obs = normalize_data(obs_seq, stats=stats['state_obs'])
            # device transfer
            nstate_obs = torch.from_numpy(nstate_obs).to(device, dtype=torch.float32)
        else:
            images = np.stack([x['image'] for x in obs_deque])
            state_obses = np.stack([x['state_obs'] for x in obs_deque])
            # normalize observation
            nstate_obs = normalize_data(state_obses, stats=stats['state_obs'])
            nimages = images
            nimages = torch.from_numpy(nimages).float().to(device, dtype=torch.float32)
            nstate_obs = torch.from_numpy(nstate_obs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            if config['type'] == 'image':
                image_features = ema_nets['vision_encoder'](nimages)
                obs_features = torch.cat([image_features, nstate_obs], dim=-1)
            else:
                obs_features = nstate_obs

            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
            if args.use_goal:
                # Goal should be the same across different time steps as in same environment
                # NOTE: Should normalize data
                goal = torch.from_numpy(info['goal_pose']).float()
                goal = normalize_data(goal, stats=stats['goal_pose']).to(device)
                obs_cond = torch.cat([obs_cond, goal.unsqueeze(0)], dim=-1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters_test)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise) # TODO: Skip steps for DDIM?
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