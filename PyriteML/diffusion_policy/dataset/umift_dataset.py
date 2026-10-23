import copy
from typing import Dict

import numpy as np
import torch
import zarr
from einops import rearrange
from PyriteConfig.tasks.umift.umift_type_conversions import (
    action9_to_action_sample,
    action10_to_action_sample,
    action19_to_action_sample,
    action21_to_action_sample,
    obs_to_obs_sample,
    raw_to_action9,
    raw_to_action10,
    raw_to_action19,
    raw_to_action21,
    raw_to_obs,
)
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler_umift import SequenceSamplerUmiFT, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

register_codecs()


class UmiFTDataset(BaseImageDataset):
    """
    hack_linear_interpolated_dense_action: if true, dense action is computed as linear interpolation of sparse actions. This effectively makes dense force/pos inputs useless.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        sparse_query_frequency_down_sample_steps: int = 1,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        hack_linear_interpolated_dense_action: bool = False,
        normalize_wrench: bool = False,
        weighted_sampling: int = 1,
        correction_horizon: int = 1,
        action_gap_threshold_mult: float = 3.0,
        scan_pre_first_fill: str = "first",
        max_abs_pos_m: float = None,
    ):
        # scan_pre_first_fill: "first" (legacy: use scan 0 before the first scan completes)
        #   or "zeros" (matches the robot runner, which feeds zeros until the first scan).
        # max_abs_pos_m: drop queries near corrupt poses (|pos| > this, in the dataset
        #   frame). Use with absolute-pose experiments (see SequenceSamplerUmiFT).
        self.scan_pre_first_fill = scan_pre_first_fill
        self.max_abs_pos_m = max_abs_pos_m
        # load into memory store
        print("[UmiFTDataset] loading data into store from", dataset_path)
        with zarr.DirectoryStore(dataset_path) as directory_store:
            replay_buffer_raw = ReplayBuffer.copy_from_store(
                src_store=directory_store, dest_store=zarr.MemoryStore()
            )
        # convert raw to replay buffer
        print("[UmiFTDataset] raw to obs/action conversion")
        action_type = ""
        if (
            shape_meta["action"]["shape"][0] == 3
            or shape_meta["action"]["shape"][0] == 6
        ):
            action_type = "pose3"
        elif (
            shape_meta["action"]["shape"][0] == 9
            or shape_meta["action"]["shape"][0] == 18
        ):
            action_type = "pose9"
        elif (
            shape_meta["action"]["shape"][0] == 10
            or shape_meta["action"]["shape"][0] == 20
        ):
            action_type = "pose9g1"
        elif (
            shape_meta["action"]["shape"][0] == 19
            or shape_meta["action"]["shape"][0] == 38
        ):
            action_type = "pose9pose9s1"
        elif (
            shape_meta["action"]["shape"][0] == 21
            or shape_meta["action"]["shape"][0] == 42
        ):
            action_type = "pose9pose9s1g2"
        else:
            raise RuntimeError("unsupported")
        self.action_type = action_type
        self.id_list = shape_meta["id_list"]
        replay_buffer = self.raw_episodes_conversion(replay_buffer_raw, shape_meta)
        # print all the shapes in the replay buffer
        # for ep in replay_buffer["data"].keys():
        #     print(f"episode {ep}")
        #     print(replay_buffer['data'][ep]["action"])
        # train/val mask for training
        val_mask = get_val_mask(
            n_episodes=replay_buffer_raw.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        print("[UmiFTDataset] creating SequenceSamplerUmiFT.")
        if action_type == "pose9":
            action_to_action_sample = action9_to_action_sample
        elif action_type == "pose9g1":
            action_to_action_sample = action10_to_action_sample
        elif action_type == "pose9pose9s1":
            action_to_action_sample = action19_to_action_sample
        elif action_type == "pose9pose9s1g2":
            action_to_action_sample = action21_to_action_sample
        else:
            raise RuntimeError("unsupported")

        sampler = SequenceSamplerUmiFT(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
            id_list=self.id_list,
            weighted_sampling=weighted_sampling,
            correction_horizon=correction_horizon,
            action_gap_threshold_mult=action_gap_threshold_mult,
            scan_pre_first_fill=scan_pre_first_fill,
            max_abs_pos_m=max_abs_pos_m,
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.sparse_query_frequency_down_sample_steps = (
            sparse_query_frequency_down_sample_steps
        )
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.sampler = sampler
        self.action_gap_threshold_mult = action_gap_threshold_mult
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.hack_linear_interpolated_dense_action = (
            hack_linear_interpolated_dense_action
        )
        self.flag_has_dense = "dense" in shape_meta["sample"]["obs"]
        self.normalize_wrench = normalize_wrench

    def raw_episodes_conversion(
        self, replay_buffer_raw: ReplayBuffer, shape_meta: dict
    ):
        replay_buffer = dict()
        replay_buffer["data"] = dict()

        for ep in replay_buffer_raw["data"].keys():
            # iterates over episodes
            # ep: 'episode_xx'
            replay_buffer["data"][ep] = dict()
            raw_to_obs(
                replay_buffer_raw["data"][ep], replay_buffer["data"][ep], shape_meta
            )
            if self.action_type == "pose9":
                raw_to_action9(
                    replay_buffer_raw["data"][ep],
                    replay_buffer["data"][ep],
                    self.id_list,
                )
            if self.action_type == "pose9g1":
                raw_to_action10(
                    replay_buffer_raw["data"][ep],
                    replay_buffer["data"][ep],
                    self.id_list,
                )
            elif self.action_type == "pose9pose9s1":
                raw_to_action19(
                    replay_buffer_raw["data"][ep],
                    replay_buffer["data"][ep],
                    self.id_list,
                )
            elif self.action_type == "pose9pose9s1g2":
                raw_to_action21(
                    replay_buffer_raw["data"][ep],
                    replay_buffer["data"][ep],
                    self.id_list,
                )
            else:
                raise RuntimeError("unsupported")
            # print(f"episode {ep}")
            # print("raw_pose:", replay_buffer_raw['data'][ep]["ts_pose_command_0"][:])
            # print("raw_policy:", replay_buffer_raw['data'][ep]["policy_pose_command_0"][:])
            # print("processed:", replay_buffer['data'][ep]["action"])

        # meta
        replay_buffer["meta"] = replay_buffer_raw["meta"]
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        if self.action_type == "pose9":
            action_to_action_sample = action9_to_action_sample
        elif self.action_type == "pose9g1":
            action_to_action_sample = action10_to_action_sample
        elif self.action_type == "pose9pose9s1":
            action_to_action_sample = action19_to_action_sample
        elif self.action_type == "pose9pose9s1g2":
            action_to_action_sample = action21_to_action_sample
        else:
            raise RuntimeError("unsupported")

        val_set.sampler = SequenceSamplerUmiFT(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
            id_list=self.id_list,
            action_gap_threshold_mult=self.action_gap_threshold_mult,
            scan_pre_first_fill=self.scan_pre_first_fill,
            max_abs_pos_m=self.max_abs_pos_m,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> tuple:
        """Compute normalizer for each key in the dataset.
        Note: only low_dim and action are considered. Image does not need normalization.
        return: tuple of normalizers for sparse and dense obs. Dense one might be None.
        """
        sparse_normalizer = LinearNormalizer()

        # gather all data in cache
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )

        data_cache_sparse = {}
        data_cache_dense = {}
        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            # sparse obs
            for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
                if self.shape_meta["obs"][key]["type"] == "low_dim":
                    if key not in data_cache_sparse.keys():
                        data_cache_sparse[key] = []
                    data_cache_sparse[key].append(
                        copy.deepcopy(batch["obs"]["sparse"][key])
                    )
            if "action" not in data_cache_sparse.keys():
                data_cache_sparse["action"] = []
            data_cache_sparse["action"].append(copy.deepcopy(batch["action"]["sparse"]))
            # dense obs
            if self.flag_has_dense is False:
                continue
            for key in self.shape_meta["sample"]["obs"]["dense"].keys():
                if self.shape_meta["obs"][key]["type"] == "low_dim":
                    if key not in data_cache_dense.keys():
                        data_cache_dense[key] = []
                    data_cache_dense[key].append(
                        copy.deepcopy(batch["obs"]["dense"][key])
                    )
            if "action" not in data_cache_dense.keys():
                data_cache_dense["action"] = []
            data_cache_dense["action"].append(copy.deepcopy(batch["action"]["dense"]))
        self.sampler.ignore_rgb(False)

        # concatenate all data
        for key in data_cache_sparse.keys():
            # data[key] = (# batches, B, T, D)
            data_cache_sparse[key] = np.concatenate(data_cache_sparse[key])  # (B, T, D)
            if not self.temporally_independent_normalization:
                data_cache_sparse[key] = rearrange(
                    data_cache_sparse[key], "B T ... -> (B T) (...)"
                )  # (B*T, D)
        if self.flag_has_dense:
            for key in data_cache_dense.keys():
                data_cache_dense[key] = np.concatenate(data_cache_dense[key])
                if not self.temporally_independent_normalization:
                    data_cache_dense[key] = rearrange(
                        data_cache_dense[key], "B H T ... -> (B H T) (...)"
                    )

        # sparse: compute normalizer for action
        sparse_action_normalizers = list()
        print("data_cache_sparse['action']", data_cache_sparse["action"].shape)

        if self.action_type == "pose3":
            AL = 3
        elif self.action_type == "pose9":
            AL = 9
        elif self.action_type == "pose9g1":
            AL = 10
        elif self.action_type == "pose9pose9s1":
            AL = 19
        elif self.action_type == "pose9pose9s1g2":
            AL = 21
        else:
            raise RuntimeError("unsupported")

        for i in range(len(self.id_list)):
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 0 : i * AL + 3]
                    )
                )
            )  # pos
            if AL < 9:
                continue
            sparse_action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 3 : i * AL + 9]
                    )
                )
            )  # rot
            if AL == 10:
                sparse_action_normalizers.append(
                    get_range_normalizer_from_stat(
                        array_to_stats(
                            data_cache_sparse["action"][..., i * AL + 9 : i * AL + 10]
                        )
                    )
                )

            if AL < 19:
                continue
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 9 : i * AL + 12]
                    )
                )
            )  # pos
            sparse_action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 12 : i * AL + 18]
                    )
                )
            )  # rot
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 18 : i * AL + 19]
                    )
                )
            )  # stiffness
            if AL < 21:
                continue
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 19 : i * AL + 20]
                    )
                )
            )  # gripper
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 20 : i * AL + 21]
                    )
                )
            )  # grasping force

        sparse_normalizer["action"] = concatenate_normalizer(sparse_action_normalizers)

        # sparse: compute normalizer for obs
        for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
            type = self.shape_meta["obs"][key]["type"]
            if type == "low_dim":
                stat = array_to_stats(data_cache_sparse[key])
                if "eef_pos" in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif "rot_axis_angle" in key:
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                elif "wrench" in key:
                    if self.normalize_wrench:
                        this_normalizer = get_range_normalizer_from_stat(stat)
                    else:
                        this_normalizer = get_identity_normalizer_from_stat(stat)
                elif "gripper" in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif "elapsed_time" in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                else:
                    raise RuntimeError("unsupported")
                sparse_normalizer[key] = this_normalizer
            elif type == "rgb":
                sparse_normalizer[key] = get_image_identity_normalizer()
            elif type == "timestamp":
                continue
            else:
                raise RuntimeError("unsupported")

        return sparse_normalizer, None

    def __len__(self):
        return len(self.sampler)

    # @profile
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs_dict, action_array = self.sampler.sample_sequence(idx)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": dict_apply(action_array, torch.from_numpy),
        }
        return torch_data
