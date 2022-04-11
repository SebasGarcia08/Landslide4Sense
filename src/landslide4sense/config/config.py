from dataclasses import dataclass
import typing as ty


@dataclass
class AugmentationConfig:
    module: str
    transforms: ty.Optional[ty.List] = None


@dataclass
class DataConfig:
    """
    Arguments:
        dir: dataset path
        train_list: training list file
        test_list: test list file
    """

    dir: str
    train_list: str
    eval_lists_paths: ty.List[str]
    eval_names: ty.List[str]
    test_list: str
    augmentation: ty.Optional[AugmentationConfig]


@dataclass
class ModelConfig:
    """
    Arguments:
        input_size: width and height of input images
        num_classes: number of classes
        module: model module to import
        name: model name in given module
    """

    input_size: str
    num_classes: int
    module: str
    name: str


@dataclass
class EarlyStoppingConfig:
    """
    Arguments:
        patience: number of epochs with no improvement after which training will be stopped
        mode: one of {min, max}
        monitor: metric to monitor
        best_result: best result to compare with
    """

    patience: int
    mode: str
    monitor: str
    best_result: float


@dataclass
class WandBConfig:
    name: str
    tags: ty.List[str]
    project: str = "landslide4sense"
    group: ty.Optional[str] = None
    id: ty.Optional[str] = None
    job_type: ty.Optional[str] = "train"


@dataclass
class CallbacksConfig:
    early_stopping: EarlyStoppingConfig
    wandb: WandBConfig


@dataclass
class TrainConfig:
    """
    Arguments:
        batch_size: number of images in each batch
        num_workers: number of workers for multithread dataloading
        learning_rate: learning rate
        num_steps: number of training steps
        num_steps_stop: number of training steps for early stopping
        weight_decay: regularisation parameter for L2-loss
        gpu_id: gpu id in the training
        snapshot_dir: snapshot directory
        restore_from: restore from snapshot
        seed: random seed
    """

    run_name: str
    tags: ty.List[str]
    start_epoch: int
    steps_per_epoch: int
    batch_size: int
    num_workers: int
    learning_rate: float
    num_steps: int
    num_steps_stop: int
    weight_decay: float
    gpu_id: int
    snapshot_dir: str
    seed: int
    callbacks: CallbacksConfig
    restore_from: ty.Optional[str] = None


@dataclass
class Config:
    """
    Arguments:
        data: data configuration
        model: model configuration
        training: training configuration
    """

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
