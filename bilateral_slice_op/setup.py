# NOTE(brendan): partly copied from
# https://github.com/pytorch/pytorch/blob/c3e3c5cc39165470ddab5afb6373399fdbd6598e/tools/setup_helpers/cuda.py
# and
# https://github.com/pytorch/pytorch/blob/c3e3c5cc39165470ddab5afb6373399fdbd6598e/tools/setup_helpers/env.py
import ctypes.util
import glob
import os
import platform
import re
from subprocess import Popen, PIPE

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


LINUX_HOME = '/usr/local/cuda'
WINDOWS_HOME = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_negative_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']


def find_nvcc():
    if IS_WINDOWS:
        proc = Popen(['where', 'nvcc.exe'], stdout=PIPE, stderr=PIPE)
    else:
        proc = Popen(['which', 'nvcc'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        if IS_WINDOWS:
            if out.find('\r\n') != -1:
                out = out.split('\r\n')[0]
            out = os.path.abspath(os.path.join(os.path.dirname(out), ".."))
            out = out.replace('\\', '/')
            out = str(out)
        return os.path.dirname(out)
    else:
        return None


def find_cuda_version(cuda_home):
    if cuda_home is None:
        return None
    if IS_WINDOWS:
        candidate_names = [os.path.basename(cuda_home)]
    else:
        # get CUDA lib folder
        cuda_lib_dirs = ['lib64', 'lib']
        for lib_dir in cuda_lib_dirs:
            cuda_lib_path = os.path.join(cuda_home, lib_dir)
            if os.path.exists(cuda_lib_path):
                break
        # get a list of candidates for the version number
        # which are files containing cudart
        candidate_names = list(glob.glob(os.path.join(cuda_lib_path, '*cudart*')))
        candidate_names = [os.path.basename(c) for c in candidate_names]

    # suppose version is MAJOR.MINOR.PATCH, all numbers
    version_regex = re.compile(r'[0-9]+\.[0-9]+\.[0-9]+')
    candidates = [c.group() for c in map(version_regex.search, candidate_names) if c]
    if len(candidates) > 0:
        # normally only one will be retrieved, take the first result
        return candidates[0]
    # if no candidates were found, try MAJOR.MINOR
    version_regex = re.compile(r'[0-9]+\.[0-9]+')
    candidates = [c.group() for c in map(version_regex.search, candidate_names) if c]
    if len(candidates) > 0:
        return candidates[0]


if check_negative_env_flag('USE_CUDA') or check_env_flag('USE_ROCM'):
    USE_CUDA = False
    CUDA_HOME = None
    CUDA_VERSION = None
else:
    if IS_LINUX or IS_DARWIN:
        CUDA_HOME = os.getenv('CUDA_HOME', LINUX_HOME)
    else:
        CUDA_HOME = os.getenv('CUDA_PATH', '').replace('\\', '/')
        if CUDA_HOME == '' and len(WINDOWS_HOME) > 0:
            CUDA_HOME = WINDOWS_HOME[0].replace('\\', '/')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        if IS_LINUX or IS_WINDOWS:
            cuda_path = find_nvcc()
        else:
            cudart_path = ctypes.util.find_library('cudart')
            if cudart_path is not None:
                cuda_path = os.path.dirname(cudart_path)
            else:
                cuda_path = None
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
        else:
            CUDA_HOME = None
    CUDA_VERSION = find_cuda_version(CUDA_HOME)
    USE_CUDA = CUDA_HOME is not None

ext_modules = []
if USE_CUDA:
    ext_modules.append(
        CUDAExtension('bilateral_slice',
                      ['bilateral_slice.cpp',
                       'bilateral_slice_cuda_kernel.cu'],
                      extra_compile_args=['-DBISLICE_CUDA']))
else:
    ext_modules.append(
        CppExtension('bilateral_slice',
                     ['bilateral_slice.cpp', 'bilateral_slice_cpu.cpp']))

setup(name='bilateral_slice_op',
      ext_modules=ext_modules,
      cmdclass={
          'build_ext': BuildExtension
      })
