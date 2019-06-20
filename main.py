import os
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from catalyst.dl.callbacks import Callback, RunnerState
from catalyst.dl.experiments import SupervisedRunner
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst_utils import ShowPolarBatchesCallback
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy, \
    rgb_image_from_tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.linknet import LinkNet34

CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']

COLORS = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128),
          (0, 0, 192),
          (128, 128, 0), (192, 128, 128), (64, 64, 128), (64, 0, 128),
          (64, 64, 0), (0, 128, 192), (0, 0, 0)]


class CamVidDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            transform=A.Normalize()
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in
                           self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in
                          self.ids]
        self.transform = transform

    def __getitem__(self, i):
        # read data
        image = fs.read_rgb_image(self.images_fps[i])
        mask = fs.read_image_as_is(self.masks_fps[i])
        assert mask.max() < len(CLASSES)

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        return {
            "image_id": id_from_fname(self.images_fps[i]),
            "features": tensor_from_rgb_image(image),
            "targets": torch.from_numpy(mask).long()
        }

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [

        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=5, shift_limit=0.1, p=1, border_mode=cv2.BORDER_CONSTANT),
        # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.RandomSizedCrop(min_max_height=(300, 360), height=320, width=320,
                          always_apply=True),
        A.HorizontalFlip(p=0.5),

        A.IAAAdditiveGaussianNoise(p=0.2),
        # A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1),
                A.NoOp()
            ]
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
                A.NoOp()
            ]
        ),

        A.Normalize(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Normalize(),
    ]
    return A.Compose(test_transform)


def visualize_predictions(input: dict, output: dict,
                          mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)):
    images = []
    for image, target, image_id, logits in zip(input['features'],
                                               input['targets'],
                                               input['image_id'],
                                               output['logits']):
        image = rgb_image_from_tensor(image, mean, std)
        # target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits).argmax(axis=0)

        overlay = image.copy()
        for class_index, class_color in enumerate(range(len(COLORS))):
            image[logits == class_index, :] = class_color

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN,
                    1, (250, 250, 250))

        images.append(overlay)
    return images


class MulticlassIoUCallback(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits",
                 prefix: str = "iou"):
        """
        :param input_key: input key to use for IoU calculation; specifies our `y_true`.
        :param output_key: output key to use for IoU calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.eps = 1e-7

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        dtype = outputs.dtype
        batch_size = int(outputs.size(0))
        num_classes = int(outputs.size(1))

        # Binarize outputs as we don't want to compute soft-jaccard
        outputs = outputs.argmax(dim=1)

        # outputs = outputs.cpu()
        # targets = targets.cpu()

        ious_per_class = []
        for class_index in range(num_classes):
            pred_class_mask = (outputs[:, class_index, ...] == class_index).type(dtype)
            true_class_mask = (targets[:, class_index, ...] == class_index).type(dtype)

            intersection = float(torch.sum(true_class_mask * pred_class_mask))
            union = float(torch.sum(true_class_mask) + torch.sum(pred_class_mask))
            true_counts = int(torch.sum(true_class_mask) > 0)

            if true_counts:
                iou = intersection / (union - intersection + self.eps)
                ious_per_class.append(iou)
            else:
                ious_per_class.append(0)

        metric = np.mean(ious_per_class)
        state.metrics.add_batch_value(name=self.prefix, value=metric)


def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'CamVid')

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    train_ds = CamVidDataset(x_train_dir, y_train_dir,
                             transform=get_training_augmentation())
    valid_ds = CamVidDataset(x_valid_dir, y_valid_dir,
                             transform=get_validation_augmentation())

    data_loaders = OrderedDict()
    num_train_samples = len(train_ds)
    mul_factor = 10

    data_loaders['train'] = DataLoader(train_ds, batch_size=32, shuffle=False,
                                       num_workers=16, pin_memory=True,
                                       sampler=WeightedRandomSampler(np.ones(num_train_samples), num_train_samples * mul_factor))

    data_loaders['valid'] = DataLoader(valid_ds, batch_size=32, shuffle=False,
                                       num_workers=4, pin_memory=True)

    print(len(train_ds), len(valid_ds))

    num_classes = len(CLASSES)
    model = LinkNet34(num_classes).cuda()

    # model runner
    runner = SupervisedRunner()

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[30, 60, 90, 120],
                                                     gamma=0.5)

    # model training
    runner.train(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=1e-4),
        scheduler=scheduler,
        callbacks=[
            MulticlassIoUCallback(prefix='iou'),
            ShowPolarBatchesCallback(visualize_predictions, metric='loss',
                                     minimize=True),
        ],
        logdir='runs',
        loaders=data_loaders,
        num_epochs=150,
        verbose=False,
        main_metric='iou',
        minimize_metric=False
    )


if __name__ == '__main__':
    main()
