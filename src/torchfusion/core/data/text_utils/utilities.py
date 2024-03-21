from typing import TYPE_CHECKING, Tuple


def rename_key(data, renamed_key, key):
    if renamed_key in data:
        data[key] = data[renamed_key]
        del data[renamed_key]


def remove_keys(data, keys):
    for k in keys:
        if k in data:
            del data[k]


def get_bbox_center(bbox: Tuple[int, int, int, int]):
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def normalize_bbox(bbox: Tuple[int, int, int, int], size: Tuple[int, int]):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def pad_sequences(sequences, padding_side, max_length, padding_elem):
    if padding_side == "right":
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        seq + [padding_elem] * (max_length - len(seq))
                    )
            return padded_sequences
        else:
            return [seq + [padding_elem] * (max_length - len(seq)) for seq in sequences]
    else:
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        [padding_elem] * (max_length - len(seq)) + seq
                    )
            return padded_sequences
        else:
            return [[padding_elem] * (max_length - len(seq)) + seq for seq in sequences]
