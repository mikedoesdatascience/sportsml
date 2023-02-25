import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="mlp")
def train(cfg : DictConfig) -> None:
    trainer = hydra.utils.instantiate(cfg.trainer)
    model = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.dm)

    trainer.fit(model, dm)

    trainer.test(model, dm, ckpt_path='best')

if __name__ == "__main__":
    train()
