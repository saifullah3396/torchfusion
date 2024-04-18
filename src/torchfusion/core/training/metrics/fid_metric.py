import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from ignite.metrics.gan.utils import InceptionModel, _BaseInceptionMetric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from packaging.version import Version
from torch import nn

from torchfusion.core.utilities.logging import get_logger

if Version(torch.__version__) <= Version("1.7.0"):
    torch_outer = torch.ger
else:
    torch_outer = torch.outer


# wrapper class as feature_extractor
class WrapperInceptionV3(nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3
        self.logger = get_logger()
        self.warned = False

    @torch.no_grad()
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # if the image is in range -1 to 1 we convert it to 0 to 1
        if x.min() < 0 or x.max() > 1:
            if not self.warned:
                self.logger.warning(
                    f"WrapperInceptionV3 for FID computation assumes an input image is in the range -1 to 1. Converting it to 0 to 1. Actual range = [{x.min()}, {x.max()}]"
                )
                self.warned = True
            x = (x / 2 + 0.5).clamp(0, 1)

        # inception model inputs must be images in range 0 to 1
        assert (
            x.min() >= 0.0 and x.max() <= 1.0
        ), f"Input image must be in range 0 to 1. min={x.min()}, min={x.max()}"

        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


def fid_score(
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    logger = get_logger()
    try:
        import numpy as np
    except ImportError:
        raise ModuleNotFoundError("fid_score requires numpy to be installed.")

    try:
        import scipy.linalg
    except ImportError:
        raise ModuleNotFoundError("fid_score requires scipy to be installed.")

    mu1, mu2 = mu1.cpu(), mu2.cpu()
    sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.mm(sigma2), disp=False)
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        try:
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
        except ValueError as e:
            logger.warning(e)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    if not np.isfinite(covmean).all():
        tr_covmean = np.sum(
            np.sqrt(((np.diag(sigma1) * eps) * (np.diag(sigma2) * eps)) / (eps * eps))
        )

    return float(
        diff.dot(diff).item()
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2 * tr_covmean
    )


class FID(_BaseInceptionMetric):
    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        ckpt_path: Optional[str] = None,
    ) -> None:
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires numpy to be installed.")

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scipy to be installed.")

        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._ckpt_path = Path(ckpt_path)
        self._eps = 1e-6
        self._logger = get_logger()

        super(FID, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @staticmethod
    def _online_update(
        features: torch.Tensor, total: torch.Tensor, sigma: torch.Tensor
    ) -> None:
        total += features
        sigma += torch_outer(features, features)

    def _get_covariance(self, sigma: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates covariance from mean and sum of products of variables
        """

        sub_matrix = torch_outer(total, total)
        sub_matrix = sub_matrix / self._num_examples

        return (sigma - sub_matrix) / (self._num_examples - 1)

    @reinit__is_reduced
    def reset(self) -> None:
        self._train_sigma = torch.zeros(
            (self._num_features, self._num_features),
            dtype=torch.float64,
            device=self._device,
        )

        self._train_total = torch.zeros(
            self._num_features, dtype=torch.float64, device=self._device
        )

        if self._ckpt_path is not None and self._ckpt_path.exists():
            ckpt = torch.load(self._ckpt_path)
            self._test_sigma = ckpt["test_sigma"]
            self._test_total = ckpt["test_total"]
            self._test_num_examples = ckpt["num_examples"]

            assert self._test_sigma.shape == (self._num_features, self._num_features)
            assert self._test_total.shape == (self._num_features,)
        else:
            self._test_sigma = torch.zeros(
                (self._num_features, self._num_features),
                dtype=torch.float64,
                device=self._device,
            )

            self._test_total = torch.zeros(
                self._num_features, dtype=torch.float64, device=self._device
            )
        self._num_examples: int = 0

        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        # train features are the predicted features, test features refer to the real features from original dataset
        # in case they are precomputed, the test features are loaded from the file and the train features are predicted
        # by the generative model
        train, test = output
        train_features = self._extract_features(train)

        # Updates the mean and covariance for the train features
        for features in train_features:
            self._online_update(features, self._train_total, self._train_sigma)

        if self._ckpt_path is None:
            test_features = self._extract_features(test)

            if (
                train_features.shape[0] != test_features.shape[0]
                or train_features.shape[1] != test_features.shape[1]
            ):
                raise ValueError(
                    f"""
                    Number of Training Features and Testing Features should be equal ({train_features.shape} != {test_features.shape})
                    """
                )

            # Updates the mean and covariance for the test features
            for features in test_features:
                self._online_update(features, self._test_total, self._test_sigma)

        self._num_examples += train_features.shape[0]

    @sync_all_reduce(
        "_num_examples", "_test_total", "_train_total", "_test_sigma", "_train_sigma"
    )
    def compute(self) -> float:
        if self._num_examples != self._test_num_examples:
            self._logger.warning(
                f"The number of examples used in evaluation {self._num_examples} are not equal to the number of examples in dataset statistics {self._test_num_examples}."
                f"Max validation used for FID are set by args.data_loader_args.max_val_samples. "
                f"The total samples usied for original fid statistics computation are set by: args.data_args.dataset_statistics_n_samples."
            )

        fid = fid_score(
            mu1=self._train_total / self._num_examples,
            mu2=self._test_total / self._test_num_examples,
            sigma1=self._get_covariance(self._train_sigma, self._train_total),
            sigma2=self._get_covariance(self._test_sigma, self._test_total),
            eps=self._eps,
        )

        if torch.isnan(torch.tensor(fid)) or torch.isinf(torch.tensor(fid)):
            warnings.warn(
                "The product of covariance of test and train features is out of bounds."
            )

        return fid

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        # # we resize it again once for model input as FID is computed with 299, 299 size
        # inputs = resize(inputs, (299, 299))

        inputs = inputs.detach()

        if inputs.device != torch.device(self._device):
            inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._feature_extractor(inputs).to(
                self._device, dtype=torch.float64
            )
        self._check_feature_shapes(outputs)

        return outputs


class FIDSaver(_BaseInceptionMetric):
    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        ckpt_path: Optional[str] = None,
    ) -> None:
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires numpy to be installed.")

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scipy to be installed.")

        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._ckpt_path = Path(ckpt_path)
        self._eps = 1e-6

        super(FIDSaver, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @staticmethod
    def _online_update(
        features: torch.Tensor, total: torch.Tensor, sigma: torch.Tensor
    ) -> None:
        total += features
        sigma += torch_outer(features, features)

    def _get_covariance(self, sigma: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates covariance from mean and sum of products of variables
        """

        sub_matrix = torch_outer(total, total)
        sub_matrix = sub_matrix / self._num_examples

        return (sigma - sub_matrix) / (self._num_examples - 1)

    @reinit__is_reduced
    def reset(self) -> None:
        if self._ckpt_path is not None and self._ckpt_path.exists():
            ckpt = torch.load(self._ckpt_path)
            self._test_sigma = ckpt["test_sigma"]
            self._test_total = ckpt["test_total"]
        else:
            self._test_sigma = torch.zeros(
                (self._num_features, self._num_features),
                dtype=torch.float64,
                device=self._device,
            )
            self._test_total = torch.zeros(
                self._num_features, dtype=torch.float64, device=self._device
            )
        self._num_examples: int = 0

        super(FIDSaver, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        test = output

        test_features = self._extract_features(test)

        # Updates the mean and covariance for the test features
        for features in test_features:
            self._online_update(features, self._test_total, self._test_sigma)

        self._num_examples += test_features.shape[0]

    @sync_all_reduce("_num_examples", "_test_total", "_test_sigma")
    def compute(self) -> float:
        if not self._ckpt_path.parent.exists():
            self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "test_sigma": self._test_sigma,
                "test_total": self._test_total,
                "num_examples": self._num_examples,
            },
            self._ckpt_path,
        )
        return 1

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        # # we resize it again once for model input as FID is computed with 299, 299 size
        # inputs = resize(inputs, (299, 299))

        inputs = inputs.detach()

        if inputs.device != torch.device(self._device):
            inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._feature_extractor(inputs).to(
                self._device, dtype=torch.float64
            )
        self._check_feature_shapes(outputs)

        return outputs
