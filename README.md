# GPT-2 117M 模型微调

## 数据准备

数据集使用 [Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)。数据集包含了歌名、歌词等信息

## 数据处理

从数据集中剔除了非英文歌词，并在输入文本之前加入了 `"The lyrics of {name of each song}"` 字段来提示输入文本为歌词。提示字段和正文空一行。数据集最后 500 个样本为测试样本，其余为训练样本

## 训练参数

```python
epoch = 1
block_size = 1024
lr = 3e-5
batch_size = 4
```

## 测试

### BERT Score

BERT Socre 包含 precision、recall 和 F1 score 三个指标

将所有测试样本的最后一个段落删去，删去的部分为参考文本，其余部分为输入文本。输入文本前加上 `"The lyrics of {name of each song}"` 字段，最后加上换行符来提示段落结束。从 500 个测试样本中删去非英文样本和文本长度超出限制的样本，最后得到 80 个有效样本。

将测试样本分别输入原模型和微调后的模型，将输出文本剔除掉输入的部分后，然后参考文本计算 BERT Score，最后计算所有样本的 BERT Score 的平均值得到最后结果。

### PPL

将测试样本完整地输入模型计算其损失，用输入文本的长度对损失值加权平均，最后通过指数函数求负对数似然，从而得到 PPL 值。

### 结果

| Metrics | Original Model | Finetuned Model|
| --- | --- | --- |
| Precision | 0.4078 | 0.4742 |
| Recall | 0.3570 | 0.4687 |
| F1 | 0.3734 | 0.4614 |
| PPL | 23.48 | 13.97 |
