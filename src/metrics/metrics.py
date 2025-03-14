from torch import nn
from src.utils.globals import PAD_ID


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, label_smoothing):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=PAD_ID)
        self.name = "loss"

    def forward(self, logits, trg_text, **batch):
        return {"loss": self.loss(logits, trg_text)}



