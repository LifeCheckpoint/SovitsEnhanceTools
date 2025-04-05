import torch
import torch.nn as nn
import random as ran
import tqdm as tqdm
from typing import List
from auto_encoder.data import read_data
from auto_encoder.model import ConvAE

def train(tensors: List[torch.Tensor], epoch: int = 10**4, batch_size: int = 64):
    # 模型初始化
    model = ConvAE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 发送到 GPU
    if torch.cuda.is_available():
        model = model.cuda()
        tensors = [tensor.cuda() for tensor in tensors]

    # 训练
    for i in tqdm.tqdm(range(epoch), desc="自编码器训练", unit="epoch"):
        # 随机选择一个批次
        indices = ran.choices(tensors, k=batch_size) # 维度为 batch_size * (1, 256, 64)
        indices = torch.cat(indices, dim=0)

        output = model(indices)
        loss = criterion(output, indices)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            tqdm.tqdm.write(f"Epoch [{i}/{epoch}], Loss: {loss.item():.4f}")

    if torch.cuda.is_available():
        model = model.cpu()
        tensors = [tensor.cpu() for tensor in tensors]

    return model