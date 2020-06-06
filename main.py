import os
from collections import OrderedDict
from datetime import datetime
from functools import partial

import numpy as np
import torch.nn as nn
from catalyst.dl import SupervisedRunner, CriterionCallback, EarlyStoppingCallback, SchedulerCallback
from pytorch_toolbelt.losses import DiceLoss
from pytorch_toolbelt.utils.catalyst import (
    IoUMetricsCallback,
    draw_semantic_segmentation_predictions,
    ShowPolarBatchesCallback,
)
from pytorch_toolbelt.utils.random import set_manual_seed
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

import camvid


def train_baseline():
    set_manual_seed(42)

    mul_factor = 10
    num_epochs = 50

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    x_train_dir = os.path.join(DATA_DIR, "train")
    y_train_dir = os.path.join(DATA_DIR, "trainannot")

    x_valid_dir = os.path.join(DATA_DIR, "val")
    y_valid_dir = os.path.join(DATA_DIR, "valannot")

    train_ds = camvid.CamVidDataset(x_train_dir, y_train_dir, transform=camvid.get_training_augmentation())
    valid_ds = camvid.CamVidDataset(x_valid_dir, y_valid_dir, transform=camvid.get_validation_augmentation())

    num_classes = len(camvid.CLASS_NAMES)

    print("Train dataset size", len(train_ds))
    print("Valid dataset size", len(valid_ds))

    # Define callbacks

    iou_score = IoUMetricsCallback(
        metric="iou",
        mode="multiclass",
        # We exclude 'unlabeled' class from the evaluation
        class_names=camvid.CLASS_NAMES[:11],
        classes_of_interest=np.arange(11),
        prefix="iou",
    )

    visualize_predictions = partial(
        draw_semantic_segmentation_predictions, mode="side-by-side", class_colors=camvid.CLASS_COLORS
    )

    show_batches = ShowPolarBatchesCallback(visualize_predictions, metric="iou", minimize=True)
    if mul_factor > 1:
        train_ds = ConcatDataset([train_ds] * mul_factor)
    loaders = OrderedDict()
    loaders["train"] = DataLoader(train_ds, batch_size=4, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    loaders["valid"] = DataLoader(valid_ds, batch_size=4, num_workers=4, pin_memory=True, drop_last=False)

    model = camvid.get_model("LinkNet34", num_classes=num_classes, dropout=0.1).cuda()

    optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5)
    early_stopping = EarlyStoppingCallback(patience=12, metric="iou", minimize=False)

    current_time = datetime.now().strftime("%b%d_%H_%M")
    prefix = f"runs/linknet34/baseline/{current_time}"

    log_dir = os.path.join("runs", prefix)
    os.makedirs(log_dir, exist_ok=False)

    # model training
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion={"ce": nn.CrossEntropyLoss(), "dice": DiceLoss(mode="multiclass", log_loss=True)},
        optimizer=optimizer,
        callbacks=[
            iou_score,
            early_stopping,
            CriterionCallback(input_key="targets", output_key="logits", criterion_key="ce"),
            CriterionCallback(input_key="targets", output_key="logits", criterion_key="dice", multiplier=0.5),
            SchedulerCallback(reduce_metric="iou"),
            show_batches,
        ],
        logdir=log_dir,
        loaders=loaders,
        num_epochs=num_epochs,
        scheduler=scheduler,
        verbose=True,
        main_metric="iou",
        minimize_metric=False,
    )


def main():
    train_baseline()


if __name__ == "__main__":
    main()
