# my_app.py
from omegaconf import ListConfig, OmegaConf


def resolve_dir_name(input: str) -> str:
    ret = input.replace("{", "")
    ret = ret.replace("}", "")
    ret = ret.replace("[", "")
    ret = ret.replace("]", "")
    ret = ret.replace(",", "_")
    ret = ret.replace("/", "_")
    ret = ret.replace("=", "-")
    return ret


def dir_name_from_overrides(overrides: ListConfig) -> str:
    task_overrides: ListConfig = overrides.task
    overrides_filtered = []
    for override in task_overrides:
        ignore = False
        for key in [
            "experiment",
            "analysis",
            "do_train",
            "do_test",
            "da_val",
            "args/data_args",
            "args/model_args",
            "n_splits",
            "n_devices"
        ]:
            if key in override:  # ignore the experiment overrides
                ignore = True
        if not ignore:
            key, value = override.split("=")
            key = key.split(".")[-1]
            overrides_filtered.append(f"{key}={value}")
    ret: str = "_".join(overrides_filtered)
    ret = ret.replace("{", "")
    ret = ret.replace("}", "")
    ret = ret.replace("[", "")
    ret = ret.replace("]", "")
    ret = ret.replace(",", "_")
    ret = ret.replace("/", "_")
    ret = ret.replace("=", "-")
    if ret == "":
        return "default"
    return ret

def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("resolve_dir_name", resolve_dir_name)
OmegaConf.register_new_resolver("dir_name_from_overrides", dir_name_from_overrides)
OmegaConf.register_new_resolver("as_tuple", resolve_tuple)
