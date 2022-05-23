from dataclasses import dataclass
import typing as ty


@dataclass
class AugmentationConfig:
    module: str
    name: str


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
    input_size: ty.Tuple[int, int]
    eval_lists_paths: ty.List[str]
    eval_names: ty.List[str]
    test_list: str
    augmentation: ty.Optional[AugmentationConfig] = None


@dataclass
class ModelConfig:
    """
    Arguments:
        input_size: width and height of input images
        num_classes: number of classes
        module: model module to import
        name: model name in given module
    """

    module: str
    name: str
    args: ty.Dict
    restore_from: ty.Optional[str] = None


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
    name: ty.Optional[str] = None
    tags: ty.Optional[ty.List[str]] = None
    project: str = "landslide4sense"
    group: ty.Optional[str] = None
    id: ty.Optional[str] = None
    job_type: ty.Optional[str] = "train"
    resume: ty.Optional[bool] = None


@dataclass
class CallbacksConfig:
    early_stopping: EarlyStoppingConfig
    wandb: WandBConfig


@dataclass
class OptimizerConfig:
    name: str
    module: str
    args: ty.Dict
    restore_from: ty.Optional[str] = None


@dataclass
class LossConfig:
    module: str
    name: str
    args: ty.Dict


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
    num_steps: int
    num_steps_stop: int
    gpu_id: int
    snapshot_dir: str
    seed: int
    callbacks: CallbacksConfig
    results_filename: str


@dataclass
class PredictConfig:
    snapshot_dir: str
    threshold: float


@dataclass
class CalibrateConfig:
    data_path: str
    num_thresholds: int
    model_filename: str


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
    optimizer: OptimizerConfig
    loss: LossConfig
    train: TrainConfig
    calibrate: CalibrateConfig
    predict: PredictConfig
