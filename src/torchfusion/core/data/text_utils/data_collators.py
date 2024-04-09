from dataclasses import dataclass, field

import torch


@dataclass
class SequenceDataCollator:
    data_key_type_map: dict = field(default_factory=lambda: {})

    def __call__(self, features):
        batch = {}

        for k, dtype in self.data_key_type_map.items():
            if k not in features[0]:
                continue
            if isinstance(features[0][k], torch.Tensor):
                batch[k] = torch.stack([sample[k] for sample in features]).type(dtype)
            elif isinstance(features[0][k], list):
                batch[k] = torch.tensor([sample[k] for sample in features], dtype=dtype)
            elif isinstance(features[0][k], str):
                batch[k] = [sample[k] for sample in features]
            else:
                batch[k] = torch.tensor([sample[k] for sample in features], dtype=dtype)
        return batch

        # if self.data_args.data_tokenizer_args:
        #     # generate overflow sample ids
        #     batch[DataKeys.OVERFLOW_MAPPING] = []
        #     for idx, token_ids in enumerate(batch[DataKeys.TOKEN_IDS]):
        #         for _ in range(len(token_ids)):
        #             batch[DataKeys.OVERFLOW_MAPPING].append(idx)
        #     batch[DataKeys.OVERFLOW_MAPPING] = torch.tensor(batch[DataKeys.OVERFLOW_MAPPING])

        #     # generate overflow token mapping
        #     overflow_to_sample_matrix = torch.zeros(
        #         len(batch["overflow_to_sample_mapping"]),
        #         batch["overflow_to_sample_mapping"].max() + 1,
        #     ).scatter_(1, batch["overflow_to_sample_mapping"].unsqueeze(1), 1.0)
        #     overflow_to_sample_matrix = torch.nn.functional.normalize(overflow_to_sample_matrix.T, p=1, dim=1)
        #     batch["overflow_to_sample_matrix"] = overflow_to_sample_matrix

        return batch
