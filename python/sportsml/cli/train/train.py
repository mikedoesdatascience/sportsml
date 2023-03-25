import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def train(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)
    trainer = hydra.utils.instantiate(cfg.trainer)
    model = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.dm)

    trainer.fit(model, dm)

    if len(dm.test_ds):
        trainer.test(model, dm, ckpt_path="best")


if __name__ == "__main__":
    train()
