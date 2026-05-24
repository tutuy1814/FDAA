"""
日志工具模块

包含:
1. 日志配置
2. AverageMeter
3. 进度跟踪
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    配置日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 日志级别
        format_str: 日志格式
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的handlers
    logger.handlers.clear()

    formatter = logging.Formatter(format_str)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    显示训练进度
    """
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int, logger: Optional[logging.Logger] = None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Timer:
    """
    计时器
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.lap_time = self.start_time

    def elapsed(self) -> float:
        """返回从开始到现在的时间"""
        return time.time() - self.start_time

    def lap(self) -> float:
        """返回从上次lap到现在的时间"""
        current_time = time.time()
        elapsed = current_time - self.lap_time
        self.lap_time = current_time
        return elapsed


class MetricLogger:
    """
    记录和可视化训练指标
    """
    def __init__(self, delimiter: str = "\t"):
        self.meters: Dict[str, AverageMeter] = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ':.4f')
            self.meters[k].update(v)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)

    def get_avg_dict(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}


# 用于导入torch时不报错
try:
    import torch
except ImportError:
    pass


# 测试代码
if __name__ == "__main__":
    print("Testing logger...")

    # 测试日志
    logger = setup_logger("test", level=logging.DEBUG)
    logger.info("This is an info message")
    logger.debug("This is a debug message")

    # 测试AverageMeter
    meter = AverageMeter('loss', ':.4f')
    for i in range(10):
        meter.update(0.1 * i)
    print(f"Average: {meter.avg:.4f}")

    # 测试Timer
    timer = Timer()
    time.sleep(0.1)
    print(f"Elapsed: {timer.elapsed():.2f}s")

    print("Logger test passed!")
