import dataclasses
import logging

from torchfusion.core.data.data_modules.fusion_data_module import FusionDataModule
from torchfusion.core.data.factory.data_augmentation import DataAugmentationFactory
from torchfusion.core.data.factory.train_val_sampler import TrainValSamplerFactory
from torchfusion.core.data.utilities.containers import TransformsDict
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import print_transforms
from torchfusion.core.utilities.logging import get_logger
from torchvision.transforms import Compose

logger = get_logger(__name__)


def load_transforms_from_config(train_augs, eval_augs):
    # define data transforms according to the configuration
    tf = TransformsDict()
    if train_augs is not None:
        tf.train = []
        for aug_args in train_augs:
            aug = DataAugmentationFactory.create(
                aug_args.name,
                aug_args.kwargs,
            )
            tf.train.append(aug)

    if eval_augs is not None:
        tf.validation = []
        tf.test = []
        for aug_args in eval_augs:
            aug = DataAugmentationFactory.create(
                aug_args.name,
                aug_args.kwargs,
            )
            tf.validation.append(aug)
            tf.test.append(aug)

    # wrap the transforms in a callable class
    if tf.train is not None:
        tf.train = Compose(tf.train)

    if tf.validation is not None:
        tf.validation = Compose(tf.validation)

    if tf.test is not None:
        tf.test = Compose(tf.test)

    return tf


def load_datamodule_from_args(
    args,
    stage: TrainingStage = TrainingStage.train,
    preprocess_transforms=None,
    realtime_transforms=None,
    override_collate_fns=None,
    rank=0,
) -> FusionDataModule:
    """
    Initializes the datamodule for training.
    """

    import ignite.distributed as idist
    from torchfusion.core.data.data_modules.fusion_data_module import FusionDataModule

    logger.info("Setting up datamodule...")

    # setup transforms
    preprocess_transforms_from_config = load_transforms_from_config(
        args.train_preprocess_augs,
        args.eval_preprocess_augs,
    )
    realtime_transforms_from_config = load_transforms_from_config(
        args.train_realtime_augs,
        args.eval_realtime_augs,
    )

    # load from config or override
    preprocess_transforms = (
        preprocess_transforms
        if preprocess_transforms is not None
        else preprocess_transforms_from_config
    )
    realtime_transforms = (
        realtime_transforms
        if realtime_transforms is not None
        else realtime_transforms_from_config
    )

    # print transforms
    print_transforms(
        preprocess_transforms,
        title="preprocess transforms",
        log_level=logging.INFO,
    )
    print_transforms(
        realtime_transforms,
        title="realtime transforms",
        log_level=logging.INFO,
    )

    # setup train_val_sampler
    train_val_sampler = None
    if (
        args.general_args.do_val
        and not args.data_loader_args.use_test_set_for_val
        and args.train_val_sampler is not None
    ):
        # setup train/val sampler
        train_val_sampler = TrainValSamplerFactory.create(
            args.train_val_sampler.name,
            args.train_val_sampler.kwargs,
        )

    # initialize data module generator function
    collate_fn_kwargs = {}
    if override_collate_fns:
        collate_fn_kwargs = dict(collate_fns=override_collate_fns)
    # else:
    #     # load default collate fns
    #     if args.data_loader_args.default_collate_fn_type == 'standard':
    #         default_collate_fns = CollateFnDict(train=BatchToTensorDataCollator(), validation=BatchToTensorDataCollator(), test=BatchToTensorDataCollator())
    #         collate_fn_kwargs = dict(collate_fns=default_collate_fns)
    #     elif args.data_loader_args.default_collate_fn_type == 'sequence':
    #         default_collate_fns = CollateFnDict(train=BaseSequenceDataCollator(), validation=BaseSequenceDataCollator(), test=BaseSequenceDataCollator())
    #         collate_fn_kwargs = dict(collate_fns=default_collate_fns)

    datamodule = FusionDataModule(
        dataset_name=args.data_args.dataset_name,
        dataset_cache_dir=args.data_args.dataset_cache_dir,
        dataset_dir=args.data_args.dataset_dir,
        cache_file_name=args.data_args.cache_file_name,
        use_auth_token=args.data_args.use_auth_token,
        dataset_config_name=args.data_args.dataset_config_name,
        preprocess_transforms=preprocess_transforms,
        realtime_transforms=realtime_transforms,
        train_val_sampler=train_val_sampler,
        preprocess_batch_size=args.data_args.preprocess_batch_size,
        dataset_kwargs=args.data_args.dataset_kwargs,
        tokenizer_config=(
            dataclasses.asdict(args.data_args.tokenizer_config)
            if args.data_args.tokenizer_config is not None
            else None
        ),
        num_proc=args.data_args.num_proc,
        compute_dataset_statistics=args.data_args.compute_dataset_statistics,
        dataset_statistics_n_samples=args.data_args.dataset_statistics_n_samples,
        stats_filename=args.data_args.stats_filename,
        features_path=args.data_args.features_path,
        **collate_fn_kwargs,
    )

    # only download dataset on rank 0, all other ranks wait here for rank 0 to load the datasets
    if rank > 0:
        idist.barrier()

    # we manually prepare data and call setup here so dataset related properties can be initalized.
    datamodule.setup(
        stage=stage,
        do_train=args.general_args.do_train,
        max_train_samples=args.data_loader_args.max_train_samples,
        max_val_samples=args.data_loader_args.max_val_samples,
        max_test_samples=args.data_loader_args.max_test_samples,
        use_test_set_for_val=args.data_loader_args.use_test_set_for_val,
    )

    if rank == 0:
        idist.barrier()

    return datamodule
