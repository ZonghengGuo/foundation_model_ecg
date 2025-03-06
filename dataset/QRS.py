import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import ctypes
import os
import subprocess


def compile_shared_lib():
    """
    如果共享库（例如 libqrs.so）不存在，则编译 C++ 源文件为共享库。
    需要确保系统中安装了 g++ 编译器，且 eplim.cpp 与 QRSFILT.cpp 在当前目录中。
    """
    lib_name = "libqrs.so"
    if not os.path.exists(lib_name):
        # 编译命令：-shared 表示生成共享库，-fPIC 表示生成位置无关代码，-O3 表示优化级别
        cmd = ["g++", "-shared", "-fPIC", "-O3", "-o", lib_name, "eplim.cpp", "QRSFILT.cpp"]
        print("正在编译 C++ 文件为共享库...")
        subprocess.check_call(cmd)
        print("编译成功：", lib_name)
    return lib_name


# 编译共享库（如果尚未编译）
lib_path = compile_shared_lib()
# 加载共享库，注意路径需为绝对路径
_lib = ctypes.cdll.LoadLibrary(os.path.abspath(lib_path))

# 设置 eplim 函数的参数和返回类型
# 假设 eplim 函数接口为： void eplim(const double* ecg, int n, int* qrs)
_lib.eplim.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
]
_lib.eplim.restype = None


def eplim(ecg):
    """
    调用共享库中的 eplim 函数进行 QRS 检测
    :param ecg: 经过放大后的 ECG 信号（1D numpy 数组）
    :return: 检测结果数组（非零位置表示检测到 QRS）
    """
    n = ecg.size
    qrs = np.zeros(n, dtype=np.int32)
    _lib.eplim(ecg, n, qrs)
    return qrs


def QRS(ecg, fs):
    """
    检测 ECG 信号中的 QRS 复合波，步骤包括：
      1. 重采样至 200 Hz
      2. 4 阶 Butterworth 带通滤波（8~20 Hz）
      3. 信号扩展（Orlo）：在前端添加 1500 个样本，以减小滤波边缘效应
      4. 将信号幅值放大 10000 倍后调用 C++ 实现的 eplim 函数
      5. 后处理：提取检测到的 QRS 索引，并补偿因 Orlo 引入的偏移
    :param ecg: 原始 ECG 信号（1D numpy 数组）
    :param fs: 原始采样率
    :return: QRS 检测结果索引（在重采样后信号中）
    """
    # 目标采样率
    fc_ecg = 200

    # 1. 重采样：使用 resample_poly 将信号从 fs 重采样到 fc_ecg
    ecg_resampled = resample_poly(ecg, fc_ecg, int(fs))

    # 2. 带通滤波：4 阶 Butterworth 带通滤波器（通带：8~20 Hz）
    nyq = fc_ecg / 2.0
    low = 8 / nyq
    high = 20 / nyq
    b, a = butter(4, [low, high], btype='bandpass')
    ecg_filtered = filtfilt(b, a, ecg_resampled)

    # 3. Orlo 操作：在信号前端添加 1500 个样本以避免滤波边缘效应
    ecg_extended = np.concatenate((ecg_filtered[:1500], ecg_filtered))

    # 4. 调用 C++ eplim 函数前，放大信号幅值（乘以 10000）
    scaled_ecg = ecg_extended * 10000
    qrs_result = eplim(scaled_ecg)

    # 5. 后处理：提取非零输出的索引，并扣除前面添加的 1500 个样本
    detected_indices = np.nonzero(qrs_result)[0]
    r = detected_indices - 1500
    return r


# ---------------------------
# 示例：如何调用 QRS 函数
# ---------------------------
if __name__ == "__main__":
    # 模拟一个 ECG 信号（实际使用时请替换为真实数据）
    fs = 1000  # 原始采样率，例如 1000 Hz
    t = np.linspace(0, 10, 10 * fs, endpoint=False)
    # 此处构造一个示例信号（例如简单正弦波，不是真正的 ECG）
    ecg = np.sin(2 * np.pi * 1 * t)

    # 调用 QRS 函数检测 QRS 复合波
    r = QRS(ecg, fs)
    print("检测到的 QRS 索引:", r)
