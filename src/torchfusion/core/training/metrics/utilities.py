from typing import Any, Callable, Mapping, Sequence, Union, cast

import ignite.distributed as idist
import torch
from ignite.metrics.metric import Metric, reinit__is_reduced


def output_transform(x: Any, index: int, name: str) -> Any:
    import numbers

    import torch

    if isinstance(x, Mapping):
        return x[name]
    elif isinstance(x, Sequence):
        return x[index]
    elif isinstance(x, (torch.Tensor, numbers.Number)):
        return x
    else:
        raise TypeError(
            "Unhandled type of update_function's output. " f"It should either mapping or sequence, but given {type(x)}"
        )


class OutputGatherer(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(OutputGatherer, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._outputs = []

    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:
        output = output.clone().to(self._device)
        self._outputs.append(output)

    def compute(self) -> Any:
        if len(self._outputs[0].shape) == 0:
            _outputs_tensor = torch.tensor(self._outputs)
        else:
            _outputs_tensor = torch.cat(self._outputs, dim=0)

        ws = idist.get_world_size()

        if ws > 1 and not self._is_reduced:
            # All gather across all processes
            _outputs_tensor = cast(torch.Tensor, idist.all_gather(_outputs_tensor))
        self._is_reduced = True
        return _outputs_tensor
