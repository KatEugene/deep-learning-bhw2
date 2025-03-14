import torch
from src.utils.globals import ROOT_PATH
from src.trainer.trainer import Trainer
from src.trainer.inferencer import Inferencer
from src.utils.helpers import set_random_seed
from src.utils.preprocessing import get_dataloaders
from src.utils.globals import config
from src.model import LanguageModel
from src.metrics import CrossEntropyLossWrapper
from src.scheduler import WarmUpScheduler
from src.metrics import WandbTracker


def train():
    set_random_seed(config.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders, decoders = get_dataloaders(config)
    model = LanguageModel(**config.model, decoders=decoders).to(device)
    criterion = CrossEntropyLossWrapper(**config.loss_function)
    optimizer = torch.optim.Adam(**config.optimizer, params=model.parameters())
    lr_scheduler = WarmUpScheduler(**config.lr_scheduler, optimizer=optimizer, epoch_len=len(dataloaders["train"]))
    wandb_tracker = WandbTracker(**config.wandb_tracker, project_config=config)
    wandb_tracker.log_config(ROOT_PATH / "src/utils/globals.py")
    wandb_tracker.log_config(ROOT_PATH / "src/model/transformer.py")
    trainer = Trainer(
        config=config,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloaders=dataloaders,
        wandb_tracker=wandb_tracker,
    )
    trainer.train()


def inference():
    set_random_seed(config.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders, decoders = get_dataloaders(config)
    test_dataloader = dataloaders["test"]
    model = LanguageModel(**config.model, decoders=decoders).to(device)

    inferencer = Inferencer(
        config=config,
        device=device,
        model=model,
        test_dataloder=test_dataloader,
    )
    inferencer.inference()


if __name__ == "__main__":
    train()
    inference()
