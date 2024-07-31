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

class FullModelHFCheckpointerSaveSteps(FullModelHFCheckpointer):
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
    
