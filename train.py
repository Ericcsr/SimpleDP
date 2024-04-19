#@markdown ### **Network Demo**
import os
import time
import numpy as np
import torch
import torch.nn as nn
import json

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from neurals import ConditionalUnet1D, create_vision_encoder
from datasets import PushTStateDataset, PushTImageDataset
from argparse import ArgumentParser


#@markdown ### **Dataset Demo**

# download demonstration data from Google Drive
parser = ArgumentParser()
parser.add_argument("--config", type=str, default="push_t_state")
parser.add_argument("--use_goal", action='store_true', default=False)
parser.add_argument("--num_epochs", type=int, default=100)
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
dataset_path = config['dataset_path']

if not os.path.isfile(dataset_path):
    print("Dataset not exists.",dataset_path)
    exit(-1)

#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
if config['type'] == 'state':
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_goal=args.use_goal
    )
    valid_dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_goal=args.use_goal,
        split="val")
else:
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_goal=args.use_goal)
    valid_dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_goal=args.use_goal,
        split="val")
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256 if config['type'] == 'state' else 64, # can be larger if more memory available
    num_workers=1 if config['type'] == 'state' else 4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=256 if config['type'] == 'state' else 64, # can be larger if more memory available
    num_workers=1 if config['type'] == 'state' else 4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon + (config['goal_dim'] if args.use_goal else 0)
)

# Create image encoder if vision
if config['type'] == 'image':
    vision_encoder = create_vision_encoder()
    nets = nn.ModuleDict({"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net})
else:
    nets = nn.ModuleDict({"noise_pred_net": noise_pred_net})

# for this demo, we use DDPMScheduler with 100 diffusion iterations

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
elif config['scheduler'] == "ddim": # TODO: parameter tuning.
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )


# device transfer
device = torch.device('cuda')
nets.to(device)

#@markdown ### **Training**
#@markdown
#@markdown Takes about an hour. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

num_epochs = args.num_epochs

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)



for epoch_idx in range(num_epochs):
    epoch_loss = list()
    # batch loop
    for i,nbatch in enumerate(dataloader):
        # data normalized in dataset
        # device transfer
        if config['type'] == 'state':
            nstate_obs = nbatch['state_obs'].to(device)
        else:
            nimage = nbatch['image'][:,:obs_horizon].float().to(device) # prevent bug from customized dataset
            nstate_obs = nbatch['state_obs'][:,:obs_horizon].to(device)
        naction = nbatch['action'].to(device)
        B = nstate_obs.shape[0]

        if config['type'] == 'image':
            image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
            image_features = image_features.reshape(*nimage.shape[:2], -1)
            obs_features = torch.cat([image_features, nstate_obs], dim=-1)
        else:
            obs_features = nstate_obs[:,:obs_horizon,:]
        # (B, obs_horizon * obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)
        if args.use_goal:
            goal = nbatch['goal'][:,0].to(device) # Use first frame goal, should be the same across entire sequence.
            obs_cond = torch.cat([obs_cond, goal], dim=-1)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(
            naction, noise, timesteps)

        # predict the noise residual
        noise_pred = nets['noise_pred_net'](
            noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(nets.parameters())

        # logging
        loss_cpu = loss.item()
        epoch_loss.append(loss_cpu)
    print("Epoch:",epoch_idx, "Loss:", np.mean(epoch_loss))
    # TODO: validate if needed
    if epoch_idx % 10 == 0 and epoch_idx > 0:
        valid_loss = list()
        with torch.no_grad():
                for i,nbatch in enumerate(valid_dataloader):
                    # data normalized in dataset
                    # device transfer
                    if config['type'] == 'state':
                        nstate_obs = nbatch['state_obs'].to(device)
                    else:
                        nimage = nbatch['image'][:,:obs_horizon].float().to(device) # prevent bug from customized dataset
                        nstate_obs = nbatch['state_obs'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nstate_obs.shape[0]

                    if config['type'] == 'image':
                        image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                        image_features = image_features.reshape(*nimage.shape[:2], -1)
                        obs_features = torch.cat([image_features, nstate_obs], dim=-1)
                    else:
                        obs_features = nstate_obs[:,:obs_horizon,:]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_features.flatten(start_dim=1)
                    if args.use_goal:
                        goal = nbatch['goal'][:,0].to(device) # Use first frame goal, should be the same across entire sequence.
                        obs_cond = torch.cat([obs_cond, goal], dim=-1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    valid_loss.append(loss.item())
        print("Validation Loss:", np.mean(valid_loss))

        

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())

# Save checkpoint
state_dict = ema_nets.state_dict()
if args.use_goal:
    torch.save(state_dict, f"checkpoints/{time.time()}_{config['type']}_goal.pth")
    torch.save(state_dict, f"checkpoints/latest_{config['type']}_goal.pth")
else:
    torch.save(state_dict, f"checkpoints/{time.time()}_{config['type']}.pth")
    torch.save(state_dict, f"checkpoints/latest_{config['type']}.pth")
