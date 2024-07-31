import gc
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch
from safetensors.torch import save_file
from torchtune import utils

from torchtune.models import convert_weights
from torchtune.models.phi3 import phi3_hf_to_tune, phi3_tune_to_hf
from torchtune.modules.rlhf.utils import reward_hf_to_tune, reward_tune_to_hf
from torchtune.utils._checkpointing._checkpointer_utils import (
    get_path,
    ModelType,
    safe_torch_load,
    save_config,
)
from torchtune.utils.logging import get_logger

from torchtune.utils._checkpointing._checkpointer import FullModelHFCheckpointer
from typing_extensions import override


log = get_logger("DEBUG")
class FullModelHFCheckpointerSaveSteps(FullModelHFCheckpointer):
    """
    Note: This is a Wrapper class around FullModelHFCheckpointer to save checkpoints at every steps that i implement myself.


    Checkpointer which reads and writes checkpoints in HF's format. For LoRA models this includes
    saving checkpoints in a format that can be loaded into PEFT via e.g. ``from_pretrained``. Examples include
    the Llama-2-7b-hf model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b-hf).

    Note:
        HF checkpoint names are usually ordered by ID (eg: 0001_of_0003, 0002_of_0003, etc.) To ensure \
        we read the files in the right order, we sort the checkpoint file names before reading.

    Note:
        Checkpoint conversion to and from HF's format requires access to model params which are \
        read directly from the ``config.json`` file. This helps ensure we either load the weights \
        correctly or error out in case of discrepancy between the HF checkpoint file and torchtune's \
        model implementations.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`

    Raises:
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """
    @override
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        safe_serialization: bool = False,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )
        self._resume_from_checkpoint = resume_from_checkpoint
        _, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint_step_")]
            if not checkpoints:
                raise ValueError(
                    "No checkpoint files found in the output directory. Cannot resume training."
                )
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1]))
            if self._is_rank_zero:
                log.info(f"Resuming training from latest checkpoint: {latest_checkpoint}")
            recipe_checkpoint_path = os.path.join(output_dir, latest_checkpoint)    
            self._recipe_checkpoint = Path(os.path.join(recipe_checkpoint_path, recipe_checkpoint))
            # update checkpoint_dir = output_dir+latest_checkpoint
            self._checkpoint_dir = Path(recipe_checkpoint_path)
        # Validate the checkpoint files
        self._checkpoint_paths = self._validate_hf_checkpoint_files(checkpoint_files)


        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)
        self._safe_serialization = safe_serialization

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint
        # load
        self._weight_map: Dict[str, str] = None

        # the config.json file contains model params needed for state dict conversion
        self._config = json.loads(
            Path.joinpath(self._checkpoint_dir, "config.json").read_text()
        )

        # save config.json to output_dir
        save_config(self._output_dir, self._config)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True


    @override
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        step: int = None,  
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        # Create a new directory for this checkpoint
        if step:
            checkpoint_dir = Path(self._output_dir) / f"checkpoint_step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Temporarily change the output directory
            original_output_dir = self._output_dir
            self._output_dir = checkpoint_dir
            # Call the parent class's save_checkpoint method
            super().save_checkpoint(
                state_dict,
                epoch,
                intermediate_checkpoint=intermediate_checkpoint,
                adapter_only=adapter_only
            )
            self._output_dir = original_output_dir
        else:
            super().save_checkpoint(
                state_dict,
                epoch,
                intermediate_checkpoint=intermediate_checkpoint,
                adapter_only=adapter_only
            )
    
