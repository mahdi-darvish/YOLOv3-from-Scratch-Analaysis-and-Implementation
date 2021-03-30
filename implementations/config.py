import cv2
import torch
import albumentations

from albumentations.pytorch import ToTensorV2


PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True


IMAGE_SIZE = 416
NUM_CLASSES = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4
MAP_IOU_THRESH = 0.5
CONF_THRESHOLD = 0.05
NMS_IOU_THRESH = 0.45


S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]


DATASET = 'PASCAL_VOC'
# "cuda" if torch.cuda.is_available() else
DEVICE = "cpu"
CHECKPOINT_FILE = "checkpoint.pth.tar"
LABEL_DIR = DATASET + "/labels/"
IMG_DIR = DATASET + "/images/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


scale = 1.1
train_transforms = albumentations.Compose(
    [
        albumentations.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        albumentations.PadIfNeeded(
            min_width=int(IMAGE_SIZE * scale),
            min_height=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        albumentations.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),

        albumentations.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),

        albumentations.OneOf(
            [
                albumentations.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                albumentations.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Blur(p=0.1),
        albumentations.Posterize(p=0.1),
        albumentations.CLAHE(p=0.1),
        albumentations.ToGray(p=0.1),
        albumentations.ChannelShuffle(p=0.05),
        albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=albumentations.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = albumentations.Compose(
    [
        albumentations.LongestMaxSize(max_size=IMAGE_SIZE),
        albumentations.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=albumentations.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),


COCO_LABELS = [
    'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]


PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]