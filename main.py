import os
from collections import OrderedDict
from functools import partial
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from catalyst.contrib.schedulers import MultiStepLR
from catalyst.dl import SupervisedRunner
from catalyst.utils import unpack_checkpoint, load_checkpoint
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.optimization.functional import get_lr_decay_parameters
from pytorch_toolbelt.losses import JointLoss, MulticlassDiceLoss, FocalLoss
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import *
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, set_trainable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet34
from datetime import datetime

CLASS_NAMES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

CLASS_COLORS = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192),
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
        assert mask.max() < len(CLASS_NAMES)

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

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)
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

        f5 = unpad_image_tensor(f5, pad)
        return f5


def get_training_augmentation():
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(300, 360), height=320, width=320, always_apply=True),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.CLAHE(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(),
            A.NoOp()
        ]),

        A.OneOf([
            A.IAASharpen(),
            A.Blur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
            A.NoOp()
        ]),

        A.OneOf([
            A.RandomFog(),
            A.RandomSunFlare(src_radius=100),
            A.RandomRain(),
            A.RandomSnow(),
            A.NoOp()
        ]),

        A.Cutout(),
        A.Normalize(),
    ])


def get_validation_augmentation():
    return A.Compose([
        A.Normalize()
    ])


def main():
    set_trainable()
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'CamVid')

    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    train_ds = CamVidDataset(x_train_dir, y_train_dir, transform=get_training_augmentation())
    valid_ds = CamVidDataset(x_valid_dir, y_valid_dir, transform=get_validation_augmentation())

    mul_factor = 10

    data_loaders = OrderedDict()
    data_loaders['train'] = DataLoader(train_ds, batch_size=4, num_workers=4, pin_memory=True, drop_last=True,
                                       sampler=WeightedRandomSampler(np.ones(len(train_ds)), len(train_ds) * mul_factor))
    data_loaders['valid'] = DataLoader(valid_ds, batch_size=4, num_workers=0, pin_memory=True, drop_last=False)

    num_classes = len(CLASS_NAMES)
    model = LinkNet34(num_classes).cuda()

    num_epochs = 50
    parameters = model.parameters()
    # parameters = get_lr_decay_parameters(model.named_parameters(),
    #                                      1e-4,
    #                                      {
    #                                          "firstconv": 0.1,
    #                                          "firstbn": 0.1,
    #                                          "encoder1": 0.1,
    #                                          "encoder2": 0.2,
    #                                          "encoder3": 0.3,
    #                                          "encoder4": 0.4,
    #                                      })
    # print(parameters)

    adam = Adam(parameters, lr=1e-4)
    # one_cycle = OneCycleLR(adam,
    #                        init_lr=1e-6,
    #                        lr_range=(1e-2, 1e-5),
    #                        num_steps=num_epochs)
    multistep = MultiStepLR(adam, [10, 20, 30, 40], gamma=0.5)

    # model runner
    runner = SupervisedRunner()

    # model training
    visualize_predictions = partial(draw_semantic_segmentation_predictions,
                                    mode='side-by-side',
                                    class_colors=CLASS_COLORS)

    current_time = datetime.now().strftime('%b%d_%H_%M')
    prefix = f'{current_time}_linknet34'

    log_dir = os.path.join('runs', prefix)
    os.makedirs(log_dir, exist_ok=False)

    runner.train(
        model=model,
        criterion=JointLoss(nn.CrossEntropyLoss(), MulticlassDiceLoss(classes=np.arange(11)), second=0.5),
        optimizer=adam,
        callbacks=[
            JaccardScoreCallback(mode='multiclass',
                                 # We exclude 'unlabeled' class from the evaluation
                                 class_names=CLASS_NAMES[:11],
                                 classes_of_interest=np.arange(11),
                                 prefix='iou'),
            ShowPolarBatchesCallback(visualize_predictions,
                                     metric='iou',
                                     minimize=True),
        ],
        logdir=log_dir,
        loaders=data_loaders,
        num_epochs=num_epochs,
        scheduler=multistep,
        verbose=True,
        main_metric='iou',
        minimize_metric=False
    )


if __name__ == '__main__':
    main()
