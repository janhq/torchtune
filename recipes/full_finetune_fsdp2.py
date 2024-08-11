# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import json
from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import init_process_group

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from torchtune import config, modules, utils
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils.activations import apply_selective_activation_checkpointing
import os
from tqdm import tqdm
import re   
from pathlib import Path
from train_utils import (
    get_num_flop_per_token,
    get_peak_flops,
    get_num_params,
)
from metrics import (
    build_gpu_memory_monitor,
)
log = utils.get_logger("DEBUG")


class FullFinetuneRecipeFSDP2(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. DDP is currently not supported. Training on CPU
            is not supported.

        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError(
                "Using fused optimizer on CPU is only supported in PyTorch nightly."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.save_every_n_steps = cfg.save_every_n_steps  # Save checkpoint every 1000 steps
        self.max_checkpoints = cfg.max_checkpoints  # Maximum number of checkpoints to keep
        # os.makedirs(self._output_dir, exist_ok=True)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self.local_step = 0

        # if seq_len is not provided in the config, set it to 4096
        self.seq_len = cfg.dataset.get("max_seq_len", 4096)
        self.checkpoint_dir = cfg.checkpointer.checkpoint_dir
        self.whole_model = None

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
            self.global_step = ckpt_dict[utils.STEPS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[utils.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[utils.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[utils.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[utils.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        ckpt_dict = self.load_checkpoint(cfg.checkpointer)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            cfg_fsdp=cfg.fsdp if hasattr(cfg, "fsdp") else None,
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=ckpt_dict[utils.OPT_KEY]
            if self._resume_from_checkpoint
            else None,
        )

        self._loss_fn = config.instantiate(cfg.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        # self.global_step += self.epochs_run * self._steps_per_epoch
        # since we are resuming from a checkpoint, we need to update the global step
        self.global_step = ckpt_dict[utils.STEPS_KEY] if self._resume_from_checkpoint else 0
        # calculate the local step within the epoch
        self.local_step = self.global_step % self._steps_per_epoch if self._resume_from_checkpoint else 0
        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_state_dict: Dict[str, Any],
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> nn.Module:
        if self._is_rank_zero:
            log.info("FSDP is enabled. Instantiating Model ...")
            init_start = time.perf_counter()

        with utils.set_default_dtype(self._dtype), torch.device("meta"):
            self.whole_model = config.instantiate(cfg_model)
            model = config.instantiate(cfg_model)
        
        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        fsdp_kwargs = {}
        if cfg_fsdp and cfg_fsdp.cpu_offload:
            from torch.distributed._composable.fsdp import CPUOffloadPolicy
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear):
                fully_shard(m, **fsdp_kwargs)
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_shard(m, **fsdp_kwargs)
            else:
                if isinstance(m, modules.TransformerDecoderLayer):
                    fully_shard(m, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        missing, unexpected = utils.load_from_full_model_state_dict(
            model, model_state_dict, self._device, self._is_rank_zero
        )

        utils.validate_no_params_on_meta_device(model)

        if self._is_rank_zero:
            log.info(
                f"Model instantiation took {time.perf_counter() - init_start:.2f} secs"
            )
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            utils.load_from_full_optimizer_state_dict(
                optimizer,
                opt_state_dict,
                self._device,
            )

        if self._is_rank_zero:
            log.info("Optimizer and loss are initialized.")
        return optimizer
    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            )
            if not packed
            else None,
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        checkpoint_dict = {}
        intermediate_checkpoint = (epoch + 1 < self.total_epochs)
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = utils.get_full_model_state_dict(
            self._model,
            self._is_rank_zero,
        )
        if intermediate_checkpoint:
            opt_state_dict = utils.get_full_optimizer_state_dict(
                self._optimizer,
                self._is_rank_zero,
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            checkpoint_dict.update({utils.MODEL_KEY: cpu_state_dict})
            
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        utils.OPT_KEY: opt_state_dict,
                        utils.SEED_KEY: self.seed,
                        utils.EPOCHS_KEY: self.epochs_run,
                        utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                        utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )
        
            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )
    def save_intermediate_checkpoint(self, epoch) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        checkpoint_dict = {}
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = utils.get_full_model_state_dict(
            self._model,
            self._is_rank_zero,
        )
        opt_state_dict = utils.get_full_optimizer_state_dict(
            self._optimizer,
            self._is_rank_zero,
        )

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            checkpoint_dict.update({utils.MODEL_KEY: cpu_state_dict})
            
            checkpoint_dict.update(
                {
                    utils.OPT_KEY: opt_state_dict,
                    utils.SEED_KEY: self.seed,
                    utils.STEPS_KEY: self.global_step,
                    utils.TOTAL_STEPS_KEY: self.total_epochs * self._steps_per_epoch,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
        
            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                step=self.global_step,
                intermediate_checkpoint=True,
            )
    def cleanup_old_checkpoints(self):
        _output_dir = self._checkpointer._output_dir
        if self._is_rank_zero:
            checkpoints = sorted(
                        [d for d in os.listdir(_output_dir) if d.startswith("checkpoint_step_")],
                        key=lambda x: int(x.split("_")[-1])
                    )
            if len(checkpoints) >= self.max_checkpoints:
                oldest_checkpoint = checkpoints.pop(0)
                oldest_checkpoint_path = Path(_output_dir) / oldest_checkpoint
                if oldest_checkpoint_path.is_dir():
                    for file in oldest_checkpoint_path.glob("*"):
                        file.unlink()
                    oldest_checkpoint_path.rmdir()
                log.info(
                    f"Deleted oldest checkpoint with path: {oldest_checkpoint_path}"
                )

    def train(self) -> None:
        """
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        # clean up before training begins
        utils.cleanup_before_training()

        _, rank = utils.get_world_size_and_rank()

        model_config = self._checkpointer._config
        # log.info(f"Model config: {model_config}")
        num_flop_per_token = get_num_flop_per_token(
            get_num_params(self.whole_model, exclude_embedding=True),
            model_config,
            self.seq_len,
        )

        # initialize GPU memory monitor and get peak flops for MFU calculation
        gpu_memory_monitor = build_gpu_memory_monitor()
        gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            # pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
            # pbar needs to start from self.local_step in case of resuming training
            pbar = tqdm(
                total=self._steps_per_epoch,
                disable=not (rank == 0),
                initial=self.local_step,
            )
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break
                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                logits = self._model(tokens, mask=mask, input_pos=input_pos)
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                # Compute loss
                loss = self._loss_fn(logits, labels)

                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):  
                        time_per_step = time.perf_counter() - t0
                        wps = num_tokens / time_per_step
                        #Copy from TorchTitan
                        # model FLOPS utilization
                        # For its definition and calculation, please refer to the PaLM paper:
                        # https://arxiv.org/abs/2204.02311
                        mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": wps,
                            "mfu": mfu
                            
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(utils.get_memory_stats(device=self._device))
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )
                    # Save checkpoint every n steps
                    if self.global_step % self.save_every_n_steps == 0:

                        self.cleanup_old_checkpoints()
                        self.save_intermediate_checkpoint(epoch=curr_epoch+1)
                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()
                    # for debugging purposes, break after 5 steps
                    # if self.global_step == 5:
                    #     break

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        torch.distributed.destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not utils.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )

    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        utils.set_torch_num_threads()

    config.log_config(recipe_name="FullFinetuneRecipeFSDP2", cfg=cfg)

    recipe = FullFinetuneRecipeFSDP2(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())