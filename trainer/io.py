import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
from urllib.parse import urlparse

import fsspec
import torch
from coqpit import Coqpit

from trainer.logger import logger


def get_user_data_dir(appname):
    if sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel, import-error

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def copy_model_files(config: Coqpit, out_path, new_fields):
    """Copy config.json and other model files to training folder and add
    new fields.

    Args:
        config (Coqpit): Coqpit config defining the training run.
        out_path (str): output path to copy the file.
        new_fields (dict): new fileds to be added or edited
            in the config file.
    """
    copy_config_path = os.path.join(out_path, "config.json")
    # add extra information fields
    new_config = {**config.to_dict(), **new_fields}
    # TODO: Revert to config.save_json() once Coqpit supports arbitrary paths.
    with fsspec.open(copy_config_path, "w", encoding="utf8") as f:
        json.dump(new_config, f, indent=4)


def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).
    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/trainer_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.
    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)


def load_checkpoint(model, checkpoint_path, use_cuda=False, eval=False):  # pylint: disable=redefined-builtin
    state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state["model"])
    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state


def save_fsspec(state: Any, path: str, **kwargs):
    """Like torch.save but can save to other locations (e.g. s3:// , gs://).

    Args:
        state: State object to save
        path: Any path or url supported by fsspec.
        **kwargs: Keyword arguments forwarded to torch.save.
    """
    with fsspec.open(path, "wb") as f:
        torch.save(state, f, **kwargs)


def save_model(config, model, optimizer, scaler, current_step, epoch, output_path, save_func, **kwargs):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    if isinstance(optimizer, list):
        optimizer_state = [optim.state_dict() for optim in optimizer]
    elif isinstance(optimizer, dict):
        optimizer_state = {k: v.state_dict() for k, v in optimizer.items()}
    else:
        optimizer_state = optimizer.state_dict() if optimizer is not None else None

    if isinstance(scaler, list):
        scaler_state = [s.state_dict() for s in scaler]
    else:
        scaler_state = scaler.state_dict() if scaler is not None else None

    if isinstance(config, Coqpit):
        config = config.to_dict()

    state = {
        "config": config,
        "model": model_state,
        "optimizer": optimizer_state,
        "scaler": scaler_state,
        "step": current_step,
        "epoch": epoch,
        "date": datetime.date.today().strftime("%B %d, %Y"),
    }
    state.update(kwargs)
    if save_func:
        save_func(state, output_path)
    else:
        save_fsspec(state, output_path)


def save_checkpoint(
    config,
    model,
    optimizer,
    scaler,
    current_step,
    epoch,
    output_folder,
    save_n_checkpoints=None,
    save_func=None,
    **kwargs,
):
    file_name = f"checkpoint_{current_step}.pth"
    checkpoint_path = os.path.join(output_folder, file_name)

    logger.info("\n > CHECKPOINT : %s", checkpoint_path)
    save_model(
        config,
        model,
        optimizer,
        scaler,
        current_step,
        epoch,
        checkpoint_path,
        save_func=save_func,
        **kwargs,
    )
    if save_n_checkpoints is not None:
        keep_n_checkpoints(output_folder, save_n_checkpoints)


def save_best_model(
    current_loss,
    best_loss,
    config,
    model,
    optimizer,
    scaler,
    current_step,
    epoch,
    out_path,
    keep_all_best=False,
    keep_after=10000,
    save_func=None,
    **kwargs,
):
    if current_loss < best_loss:
        best_model_name = f"best_model_{current_step}.pth"
        checkpoint_path = os.path.join(out_path, best_model_name)
        logger.info(" > BEST MODEL : %s", checkpoint_path)
        save_model(
            config,
            model,
            optimizer,
            scaler,
            current_step,
            epoch,
            checkpoint_path,
            model_loss=current_loss,
            save_func=save_func,
            **kwargs,
        )
        fs = fsspec.get_mapper(out_path).fs
        # only delete previous if current is saved successfully
        if not keep_all_best or (current_step < keep_after):
            model_names = fs.glob(os.path.join(out_path, "best_model*.pth"))
            for model_name in model_names:
                if os.path.basename(model_name) != best_model_name:
                    fs.rm(model_name)
        # create a shortcut which always points to the currently best model
        shortcut_name = "best_model.pth"
        shortcut_path = os.path.join(out_path, shortcut_name)
        fs.copy(checkpoint_path, shortcut_path)
        best_loss = current_loss
    return best_loss


def get_last_checkpoint(path: str) -> Tuple[str, str]:
    """Get latest checkpoint or/and best model in path.

    It is based on globbing for `*.pth` and the RegEx
    `(checkpoint|best_model)_([0-9]+)`.

    Args:
        path: Path to files to be compared.

    Raises:
        ValueError: If no checkpoint or best_model files are found.

    Returns:
        Path to the last checkpoint
        Path to best checkpoint
    """
    fs = fsspec.get_mapper(path).fs
    file_names = fs.glob(os.path.join(path, "*.pth"))
    scheme = urlparse(path).scheme
    if scheme and path.startswith(scheme + "://"):
        # scheme is not preserved in fs.glob, add it
        # back if it exists on the path
        file_names = [scheme + "://" + file_name for file_name in file_names]
    last_models = {}
    last_model_nums = {}
    for key in ["checkpoint", "best_model"]:
        last_model_num = None
        last_model = None
        # pass all the checkpoint files and find
        # the one with the largest model number suffix.
        for file_name in file_names:
            match = re.search(f"{key}_([0-9]+)", file_name)
            if match is not None:
                model_num = int(match.groups()[0])
                if last_model_num is None or model_num > last_model_num:
                    last_model_num = model_num
                    last_model = file_name

        # if there is no checkpoint found above
        # find the checkpoint with the latest
        # modification date.
        key_file_names = [fn for fn in file_names if key in fn]
        if last_model is None and len(key_file_names) > 0:
            last_model = max(key_file_names, key=os.path.getctime)
            last_model_num = load_fsspec(last_model)["step"]

        if last_model is not None:
            last_models[key] = last_model
            last_model_nums[key] = last_model_num

    # check what models were found
    if not last_models:
        raise ValueError(f"No models found in continue path {path}!")
    if "checkpoint" not in last_models:  # no checkpoint just best model
        last_models["checkpoint"] = last_models["best_model"]
    elif "best_model" not in last_models:  # no best model
        # this shouldn't happen, but let's handle it just in case
        last_models["best_model"] = last_models["checkpoint"]
    # finally check if last best model is more recent than checkpoint
    elif last_model_nums["best_model"] > last_model_nums["checkpoint"]:
        last_models["checkpoint"] = last_models["best_model"]

    return last_models["checkpoint"], last_models["best_model"]


def keep_n_checkpoints(path: str, n: int) -> None:
    """Keep only the last n checkpoints in path.

    Args:
        path: Path to files to be compared.
        n: Number of checkpoints to keep.
    """
    fs = fsspec.get_mapper(path).fs
    file_names = sort_checkpoints(path, "checkpoint")
    if len(file_names) > n:
        for file_name in file_names[:-n]:
            fs.rm(file_name)


def sort_checkpoints(output_path: str, checkpoint_prefix: str, use_mtime: bool = False) -> List[str]:
    """Sort checkpoint paths based on the checkpoint step number.

    Args:
        output_path (str): Path to directory containing checkpoints.
        checkpoint_prefix (str): Prefix of the checkpoint files.
        use_mtime (bool): If True, use modification dates to determine checkpoint order.
    """
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_path).glob(f"{checkpoint_prefix}_*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted
