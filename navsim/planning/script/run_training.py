import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "SparseDriveV2")


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        cfg=cfg,
    )
    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        cfg=cfg,
    )
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(agent=agent)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert not cfg.force_cache_computation
        assert cfg.cache_path is not None
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            test_mode=False,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            test_mode=True,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    train_data.__getitem__(10)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    # ── W&B setup ──
    logger.info("Setting up Weights & Biases tracking")
    experiment_name = cfg.get("experiment_name", "sparsedrive-training")
    agent_name = cfg.agent.get("_target_", "unknown").split(".")[-1]

    # Flatten config for wandb
    try:
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    except Exception:
        wandb_config = {}
    wandb_config["train_samples"] = len(train_data)
    wandb_config["val_samples"] = len(val_data)

    wandb_logger = WandbLogger(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=experiment_name,
        config=wandb_config,
        tags=[agent_name, cfg.get("train_test_split", {}).get("data_split", "unknown")],
        log_model=True,  # Log checkpoints as W&B artifacts
        save_dir=cfg.output_dir,
    )

    # Log anchor files as artifacts
    if hasattr(agent, "_config"):
        anchor_artifact = wandb.Artifact(f"{experiment_name}-anchors", type="anchors")
        for anchor_name in ["path_anchor", "velocity_anchor", "trajectory_anchor"]:
            anchor_path = getattr(agent._config, anchor_name, None)
            if anchor_path and Path(anchor_path).exists():
                anchor_artifact.add_file(anchor_path)
        try:
            wandb_logger.experiment.log_artifact(anchor_artifact)
        except Exception as e:
            logger.warning(f"Failed to log anchor artifact: {e}")

    # Log config yaml as artifact
    try:
        config_artifact = wandb.Artifact(f"{experiment_name}-config", type="config")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
            f.flush()
            config_artifact.add_file(f.name, name="training_config.yaml")
        wandb_logger.experiment.log_artifact(config_artifact)
        os.unlink(f.name)
    except Exception as e:
        logger.warning(f"Failed to log config artifact: {e}")

    # Watch model for gradient and parameter histograms
    wandb_logger.watch(lightning_module, log="all", log_freq=100)

    logger.info("Building Trainer")
    trainer = pl.Trainer(
        **cfg.trainer.params,
        logger=wandb_logger,
        callbacks=agent.get_training_callbacks(),
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        # val_dataloaders=val_dataloader,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
