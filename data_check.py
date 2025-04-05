from sklearn.ensemble import IsolationForest
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DATA_CHECKER
from auto_encoder.data import read_data
from auto_encoder.train import train

if __name__ == "__main__":
    # 读取数据
    tensors, file_paths = read_data(DATA_CHECKER.FILE_PATH)

    print("Warning: 重构误差仅代表模型对数据的重构能力, 不代表数据的异常程度")

    # 训练自编码器
    model = train(tensors, epoch=DATA_CHECKER.AE_TRAIN_EPOCH, batch_size=DATA_CHECKER.AE_BATCH_SIZE)

    # 计算重构误差
    recon_errors = []
    with torch.no_grad():
        for tensor in tqdm(tensors, desc="计算重构误差", unit="个"):
            tensor: torch.Tensor
            recon: torch.Tensor = model(tensor)
            recon_error = torch.mean((tensor - recon).pow(2), dim=(1,2))

            recon_errors.append(recon_error.item())

    # 寻找 TopK 最大重构误差并将对应索引文件路径输出
    recon_errors = np.array(recon_errors)
    top_k_indices = np.argsort(recon_errors)[-DATA_CHECKER.TOP_K_OUTLIERS:]
    top_k_errors = recon_errors[top_k_indices]
    top_k_paths = [file_paths[i] for i in top_k_indices]
    print(f"Top {DATA_CHECKER.TOP_K_OUTLIERS} reconstruction errors:")
    for i, (path, error) in enumerate(zip(top_k_paths, top_k_errors)):
        print(f"{i+1}: {str(path).replace('.soft.pt', '')} - {error:.4f}")

    # 离群值分数分布直方图
    plt.figure(figsize=(10, 4))
    plt.hist(recon_errors, bins=50, edgecolor='k', alpha=0.7)
    plt.axvline(
        # 阈值为 Top K 最大重构误差
        x=-np.sort(-recon_errors)[DATA_CHECKER.TOP_K_OUTLIERS], 
        color='red', 
        linestyle='--', 
        label=f'Top {DATA_CHECKER.TOP_K_OUTLIERS} outliers threshold'
    )
    plt.xlabel('Loss')
    plt.ylabel('frequency')
    plt.title('distribution of Loss')
    plt.legend()
    plt.show()
