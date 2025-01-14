import importlib
import time
import os

import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi

def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(os.path.abspath(os.path.expanduser(checkpoint_path)), map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """
    sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    noise_eps_1 = 10E-8 * np.random.randn(sample_length)
    noise_eps_2 = 10E-8 * np.random.randn(sample_length)
    if len(data_a) > sample_length:  # f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."
        frames_total = len(data_a)
        start = np.random.randint(frames_total - sample_length + 1)
        # print(f"Random crop from: {start}")
        end = start + sample_length
        return data_a[start:end]+noise_eps_1, data_b[start:end]+noise_eps_2
        # return data_a[:sample_length]+noise_eps_1, data_b[:sample_length]+noise_eps_2
    elif len(data_a) == sample_length:
        return data_a, data_b
    else:
        frames_total = len(data_a)
        return np.append(data_a, np.zeros(sample_length - frames_total)), np.append(data_b, np.zeros(
            sample_length - frames_total))


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    projection = optimal_scaling * reference
    noise = estimation - projection
    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)

def compute_SDR(clean_signal, noisy_signal):
    # return mir_eval.separation.bss_eval_sources(clean_signal, noisy_signal)[0]
    noisy_signal = noisy_signal.astype(np.float64)
    clean_signal = clean_signal.astype(np.float64)
    return si_sdr(clean_signal, noisy_signal)


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")


def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
