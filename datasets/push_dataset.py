import numpy as np
import torch
import zarr

from datasets.dataset_utils import *

# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon,
                 use_goal=False,
                 split="train"):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path)
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'state_obs': dataset_root['data']['state'][:],
            'goal_pose': dataset_root['data']['goal_pose'][:].squeeze(1) # TODO: Why need to squeeze?
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        shuffler = np.random.choice(len(indices), len(indices), replace=False)
        self.train_indices = indices[shuffler[:int(0.85*len(indices))]]
        self.val_indices = indices[shuffler[int(0.85*len(indices)):]] # No need for test set as simulation is the test
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.use_goal = use_goal
        self.split = split

    def __len__(self):
        # all possible segments of the dataset
        if self.split == "train":
            return len(self.train_indices)
        else:
            return len(self.val_indices)

    def toggle_split(self):
        if self.split == "train":
            self.split = "val"
        else:
            self.split = "train"

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.train_indices[idx] if self.split == "train" else self.val_indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['state_obs'] = nsample['state_obs'][:self.obs_horizon,:]
        if self.use_goal:
            nsample['goal'] = nsample['goal_pose'][:self.obs_horizon,:] # Should be the same across entire trajectory.
        return nsample

class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 use_goal: bool = False,\
                 split="train"):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path)

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'state_obs': dataset_root['data']['state'][:,:2],
            'action': dataset_root['data']['action'][:],
            'goal_pose': dataset_root['data']['goal_pose'][:].squeeze(1)
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        shuffler = np.random.choice(len(indices), len(indices), replace=False)
        self.train_indices = indices[shuffler[:int(0.85*len(indices))]]
        self.val_indices = indices[shuffler[int(0.85*len(indices)):]] # No need for test set as simulation is the test
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.use_goal = use_goal
        self.split = split

    def __len__(self):
        if self.split == "train":
            return len(self.train_indices)
        else:
            return len(self.val_indices)
        
    def toggle_split(self):
        if self.split == "train":
            self.split = "val"
        else:
            self.split = "train"

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.train_indices[idx] if self.split == "train" else self.val_indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['state_obs'] = nsample['state_obs'][:self.obs_horizon,:]
        if self.use_goal:
            nsample['goal'] = nsample['goal_pose'][:self.obs_horizon,:] # Should be the same across entire trajectory.
        return nsample