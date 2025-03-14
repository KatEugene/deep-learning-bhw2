from pathlib import Path
import warnings
from src.utils.helpers import ConfigDict

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_PATH = Path(__file__).parent.parent.parent

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

dict_config = {
    "globals": {
        "seed": 42,
        "max_generated_length": 100
    },
    "model": {
        "input_dim": 16000,
        "output_dim": 16000,
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_seq_len": 150,
        "max_generated_length": 100
    },
    "loss_function": {
        "label_smoothing": 0.1
    },
    "optimizer": {
        "lr": 3e-4,
        "betas": [0.9, 0.98],
        "eps": 1e-9
    },
    "dataloader": {
        "batch_size": 2,
        "max_seq_len": 150,
        "train_size": 50,
        "max_vocab_size": 16000
    },
    "lr_scheduler": {
        "d_model": 512,
        "warmup_steps": 15
    },
    "wandb_tracker": {
        "project_name": "BHW-2 DL 2024-2025",
        "run_name": "label_smoothing=0.1",
        "mode": "disabled",
        "loss_names": ["loss"],
        "run_id": None
    },
    "transforms": {
        "train": {},
        "inference": {}
    },
    "trainer": {
        "log_period": 50,
        "epoch_period": 5,
        "save_period": 5,
        "n_epochs": 1,
        "resume_from": None,
        "early_stop": None,
        "to_device": ["src_text", "trg_text"],
        "checkpoint_dir": "checkpoints"
    },
    "inferencer": {
        "save_dir": "submission",
        "from_pretrained": "default"
    }
}

config = ConfigDict(dict_config)
