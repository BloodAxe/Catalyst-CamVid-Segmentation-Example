import os
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from catalyst.dl.callbacks import IouCallback, Callback, RunnerState
from catalyst.dl.experiments import SupervisedRunner
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst_utils import ShowPolarBatchesCallback
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_mask_image, tensor_from_rgb_image, to_numpy, rgb_image_from_tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34

CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']

COLORS = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192),
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
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
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


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


def get_training_augmentation():
    train_transform = [

        # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=5, shift_limit=0.1, p=1, border_mode=cv2.BORDER_CONSTANT),
        # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.RandomSizedCrop(min_max_height=(300, 360), height=320, width=320, always_apply=True),
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
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.Normalize(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
        A.Normalize(),
    ]
    return A.Compose(test_transform)


def visualize_predictions(input: dict, output: dict, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    images = []
    for image, target, image_id, logits in zip(input['features'], input['targets'], input['image_id'], output['logits']):
        image = rgb_image_from_tensor(image, mean, std)
        # target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits).argmax(axis=0)

        overlay = image.copy()
        for class_index, class_color in enumerate(range(len(COLORS))):
            image[logits == class_index, :] = class_color

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images


class MulticlassIoUCallback(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "iou"):
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

        outputs = outputs.cpu()
        targets = targets.cpu()

        ious_per_batch = []

        for batch_index in range(batch_size):

            ious_per_class = []
            for class_index in range(num_classes):
                pred_class_mask = (outputs[batch_index] == class_index).float()
                true_class_mask = (targets[batch_index] == class_index).float()

                intersection = float(torch.sum(true_class_mask * pred_class_mask))
                union = float(torch.sum(true_class_mask) + torch.sum(pred_class_mask))
                true_counts = int(torch.sum(true_class_mask) > 0)

                if true_counts:
                    iou = intersection / (union - intersection + self.eps)
                    ious_per_class.append(iou)

            if len(ious_per_class):
                iou_per_image = np.mean(ious_per_class)
            else:
                iou_per_image = 0

            ious_per_batch.append(iou_per_image)

        metric = np.mean(ious_per_batch)
        state.metrics.add_batch_value(name=self.prefix, value=metric)


def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'CamVid')

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    train_ds = CamVidDataset(x_train_dir, y_train_dir, transform=get_training_augmentation())
    valid_ds = CamVidDataset(x_valid_dir, y_valid_dir, transform=get_validation_augmentation())

    data_loaders = OrderedDict()
    data_loaders['train'] = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    data_loaders['valid'] = DataLoader(valid_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(CLASSES)
    model = LinkNet34(num_classes).cuda()

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=1e-4),
        callbacks=[
            MulticlassIoUCallback(prefix='iou'),
            ShowPolarBatchesCallback(visualize_predictions, metric='loss', minimize=True),
        ],
        logdir='runs',
        loaders=data_loaders,
        num_epochs=50,
        verbose=False,
        main_metric='iou',
        minimize_metric=False
    )


if __name__ == '__main__':
    main()
