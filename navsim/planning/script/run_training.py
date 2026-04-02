import logging
import os
import json
import tempfile
from pathlib import Path
from typing import Tuple

import hydra
import mlflow
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


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


def _flatten_config(cfg: DictConfig, max_depth: int = 3) -> dict:
    """Flatten OmegaConf config to a flat dict for MLflow param logging.
    MLflow has a 500-param limit and 6000-char value limit."""
    flat = {}
    try:
        container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    except Exception:
        return flat

    def _recurse(obj, prefix="", depth=0):
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _recurse(v, f"{prefix}{k}.", depth + 1)
        elif isinstance(obj, (list, tuple)):
            flat[prefix.rstrip(".")] = str(obj)[:250]
        else:
            val = str(obj) if obj is not None else ""
            flat[prefix.rstrip(".")] = val[:250]

    _recurse(container)
    # MLflow limit: 500 params
    if len(flat) > 500:
        flat = dict(list(flat.items())[:500])
    return flat


def setup_mlflow(cfg: DictConfig) -> MLFlowLogger:
    """Initialize MLflow experiment tracking with SQLite backend.

    Directory structure:
        exp/mlflow/
        ├── mlflow.db          # SQLite database (metrics, params, tags, runs)
        └── artifacts/         # Artifacts (models, configs, checkpoints)
    """
    mlflow_root = Path(cfg.output_dir) / "mlflow"
    mlflow_root.mkdir(parents=True, exist_ok=True)
    (mlflow_root / "artifacts").mkdir(exist_ok=True)

    tracking_uri = cfg.get("mlflow_tracking_uri", f"sqlite:///{mlflow_root}/mlflow.db")
    artifact_location = cfg.get("mlflow_artifact_uri", str(mlflow_root / "artifacts"))
    experiment_name = cfg.get("experiment_name", "sparsedrive-training")

    mlflow.set_tracking_uri(tracking_uri)

    # Enable system metrics (GPU, CPU, memory)
    try:
        mlflow.enable_system_metrics_logging()
    except Exception:
        pass

    # Create MLFlow logger for Lightning
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        log_model=False,  # We handle model logging manually for more control
    )

    return mlf_logger


def log_experiment_context(cfg: DictConfig, agent: AbstractAgent, train_size: int, val_size: int):
    """Log comprehensive experiment context to MLflow."""
    if not mlflow.active_run():
        return

    # Tags for experiment organization
    agent_name = cfg.agent.get("_target_", "unknown").split(".")[-1]
    mlflow.set_tags({
        "agent": agent_name,
        "dataset": cfg.get("train_test_split", {}).get("data_split", "unknown"),
        "framework": "pytorch-lightning",
        "use_cache": str(cfg.use_cache_without_dataset),
    })

    # Log all hyperparameters
    params = _flatten_config(cfg)
    params["train_samples"] = train_size
    params["val_samples"] = val_size
    try:
        mlflow.log_params(params)
    except Exception as e:
        logger.warning(f"Failed to log params to MLflow: {e}")

    # Log config as artifact
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="config")
            os.unlink(f.name)
    except Exception as e:
        logger.warning(f"Failed to log config artifact: {e}")

    # Log anchor files as artifacts if they exist
    if hasattr(agent, "_config"):
        for anchor_name in ["path_anchor", "velocity_anchor", "trajectory_anchor"]:
            anchor_path = getattr(agent._config, anchor_name, None)
            if anchor_path and Path(anchor_path).exists():
                try:
                    mlflow.log_artifact(anchor_path, artifact_path="anchors")
                except Exception:
                    pass


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

    # ── MLflow setup ──
    logger.info("Setting up MLflow tracking")
    mlf_logger = setup_mlflow(cfg)

    logger.info("Building Trainer")
    trainer = pl.Trainer(
        **cfg.trainer.params,
        logger=mlf_logger,
        callbacks=agent.get_training_callbacks(),
    )

    # Log experiment context after trainer creates the MLflow run
    with mlf_logger.experiment as client:
        pass  # Ensure run is created
    log_experiment_context(cfg, agent, len(train_data), len(val_data))

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        # val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
