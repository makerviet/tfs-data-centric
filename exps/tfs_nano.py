import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 8
        self.depth = 0.33
        self.width = 0.25

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (416, 416)
        self.random_size = (14, 26)
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"

        # --------------- transform config ----------------- #
        self.mirror = 0.0
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 40
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 8.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 10
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (416, 416)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            name="train",
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val" if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
