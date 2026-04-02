import os
import logging
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent with MLflow integration."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent
        self._best_train_loss = float("inf")

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets, token = batch
        targets["token"] = token
        prediction = self.agent.forward(features, targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        for k, v in loss_dict.items():
            self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        loss = self._step(batch, "train")
        # Log learning rate
        optimizer = self.optimizers()
        if optimizer is not None:
            for i, pg in enumerate(optimizer.param_groups):
                self.log(f"train/lr_group{i}", pg["lr"], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        return self._step(batch, "val")

    def on_train_epoch_end(self):
        """Log epoch-level summary metrics to MLflow."""
        epoch = self.current_epoch
        if mlflow.active_run():
            # Log GPU memory stats if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                    mlflow.log_metrics({
                        f"gpu{i}/memory_allocated_gb": mem_alloc,
                        f"gpu{i}/memory_reserved_gb": mem_reserved,
                    }, step=epoch)

    def on_train_end(self):
        """Log final model to MLflow with artifact tracking."""
        if not mlflow.active_run():
            return

        if self.global_rank != 0:
            return

        try:
            mlflow.pytorch.log_model(
                self.agent._sparsedrive_model,
                artifact_path="final_model",
            )
            logger.info("Logged final model to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log final model to MLflow: {e}")

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        features, targets, tokens = batch
        predictions, loss_dict = self.agent.forward(features, None)
        trajectory = predictions["trajectory"]
        batch_size = trajectory.shape[0]
        results = dict()
        for i in range(batch_size):
            results[tokens[i]] = Trajectory(
                trajectory[i].cpu().numpy(),
                self.agent._config.trajectory_sampling,
            )
        return results

    def configure_optimizers(self):
        return self.agent.get_optimizers()
