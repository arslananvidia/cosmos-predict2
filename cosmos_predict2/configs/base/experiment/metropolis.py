# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# Metropolis dataset example
example_video_dataset_metropolis = L(Dataset)(
    dataset_dir="datasets/metropolis/train",
    num_frames=93,  # Standard frame count, will be cropped/padded as needed
    video_size=(740, 1280),  # Height, Width format - matches your 1280x740 videos
)

dataloader_train_metropolis = L(DataLoader)(
    dataset=example_video_dataset_metropolis,
    sampler=L(get_sampler)(dataset=example_video_dataset_metropolis),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_metropolis
predict2_video2world_training_2b_metropolis = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_metropolis",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),     # Enable EMA for better stability
                guardrail_config=dict(enabled=False),   # Disable guardrail during training
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,    # Context parallelism for handling video sequences
    ),
    dataloader_train=dataloader_train_metropolis,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),  # Report speed every 10 iterations
        ),
        max_iter=2000,  # Increased iterations for larger dataset (1,191 samples)
    ),
    checkpoint=dict(
        save_iter=500,  # Save checkpoint every 500 iterations
    ),
    optimizer=dict(
        lr=2 ** (-14.5),  # Learning rate: ~3.05e-5
    ),
    scheduler=dict(
        warm_up_steps=[2_000],      # Warm up steps
        cycle_lengths=[400_000],    # Cycle length
        f_max=[0.6],               # Maximum factor
        f_min=[0.3],               # Minimum factor
    ),
)

# Optional: 14B model configuration (requires more GPU memory)
predict2_video2world_training_14b_metropolis = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_metropolis",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=4,    # Higher context parallelism for 14B model
    ),
    dataloader_train=dataloader_train_metropolis,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=2000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[40_000],
        f_max=[0.25],
        f_min=[0.1],
    ),
)

# Register configurations
for _item in [
    # 2b, metropolis
    predict2_video2world_training_2b_metropolis,
    # 14b, metropolis (optional)
    predict2_video2world_training_14b_metropolis,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    ) 