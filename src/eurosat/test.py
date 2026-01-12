from data import create_dataloaders
import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)
@hydra.main(config_path="../../configs", config_name="default.yaml")
def main(config) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    hparams = config.experiment

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        seed=hparams["seed"],
    )

    log.info("Dataset sizes:")
    log.info("  Train batches: %d", len(train_loader))
    log.info("  Val batches:   %d", len(val_loader))
    log.info("  Test batches:  %d", len(test_loader))

    # Take one batch
    images, labels = next(iter(train_loader))

    log.info("One batch:")
    log.info("  Images shape: %s", images.shape)   # [B, 3, H, W]
    log.info("  Labels shape: %s", labels.shape)   # [B]
    log.info("  Labels dtype: %s", labels.dtype)

if __name__ == "__main__":
    main()
