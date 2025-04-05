## `tts_gen` 批量生成 GPT-Sovits 合成结果

通过读取 `references.json` 获得参考音频随机选取样本，读取 `./input` 文件夹下对应的文本后自动切分并经过 `Gradio` API 生成文本

适合制作合成数据集

## `data_check.py` 通过孤立森林检测离群数据

需要文件夹路径需要通过 `SovitsSVC` 进行数据预处理后，填写 `dataset/44k/your_character` 路径

该脚本通过 `torch.load` 读取已经预处理获得的 `.soft.pt` 提取特征张量，降维后训练一个自编码器，通过自编码器捕获数据模式的情况排查异常张量

建议通过 `vec256I9` 获取预处理的含声学特征编码的数据集

**只能检测出比较大问题的数据**，删除对应数据即可