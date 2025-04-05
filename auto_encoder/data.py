from pathlib import Path
from typing import List
from tqdm import tqdm
import torch

def read_data(file_path: str):
    file_path = Path(file_path)
    tensors = []
    file_paths = []

    # 遍历张量
    files = list(file_path.glob("*.soft.pt"))
    files.sort()  # 确保文件顺序一致（按文件名排序）
    progress_bar = tqdm(total=len(files), desc="特征张量数", unit="个")
    for file in files:
        t = torch.load(file)

        # 张量 dim 1, 2 代表特征维度与时域维度，大小较大，池化以降维
        t = torch.nn.functional.adaptive_avg_pool2d(t, (256, 64))

        tensors.append(t)
        file_paths.append(file)  # 记录当前张量对应的文件路径
        progress_bar.update()

    return tensors, file_paths