import logging
from pathlib import Path

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Save periodic checkpoints and log them to MLflow as artifacts."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.strategy.barrier()
        epoch = trainer.current_epoch
        ckpt_dir = Path(trainer.default_root_dir) / "periodic_pdm_ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"ep{epoch+1:04d}.ckpt"

        trainer.save_checkpoint(str(ckpt_path))
        trainer.print(f"[PDM] saved ckpt: {ckpt_path}")

        # Log checkpoint to MLflow
        if trainer.is_global_zero and mlflow.active_run():
            try:
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
            except Exception as e:
                logger.warning(f"Failed to log checkpoint to MLflow: {e}")

        trainer.strategy.barrier()
