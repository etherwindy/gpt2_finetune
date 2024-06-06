# GPT-2 117M 模型微调

## 数据准备

数据集使用 [Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)。数据集包含了歌名、歌词等信息。将数据集中的 `lyrics-data.csv` 文件放在 `data/lyrics/` 文件夹下。

## 数据处理

从数据集中剔除了非英文歌词，并在输入文本之前加入了 `"The lyrics of {name of each song}"` 字段来提示输入文本为歌词。提示字段和正文空一行。数据集最后 500 个样本为测试样本，其余为训练样本。

## 训练参数

```python
epoch = 1
block_size = 1024
lr = 3e-5
batch_size = 4
```

使用 4090 单卡进行训练大概需要三个小时。

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

## 生成

将输入文本写入 `input.txt`，运行 `generate.py` 即可。

更换 GPT-2 分词器使其能够应用于中文

GPT-2 原本GPT2Tokenizer，是以字节为单位的字节对编码，而不是以中文的字或词为单位。对于英文，GPT2Tokenizer大部分时候是以单词为单位进行切分的，但是对中文则完全不同，有时候2个id代表一个中文字，有时候又是1个。这一奇怪的现象正是因为采用字节对编码的结果。

考虑到上述问题，改用 BERTTokenizer 进行分词。BERTTokenizer 会先根据词表将汉字和标点符号逐个分隔，再转换为词嵌入向量。

我们使用预训练过的 bert-base-chinese 模型的分词器进行分词，模型使用原版的 GPT-2 模型，在 [People&#x27;s Daily News | Kaggle](https://www.kaggle.com/datasets/concyclics/renmindaily) 数据集上微调，使 GPT-2 模型能够理解和生成中文。

训练：

```python
python finetune_chinese.py
```

生成：

```python
python generate_chinese.py
```

生成时的输入文本在代码文件中修改。

## GPT-2 中文数据集微调

使用预训练的 gpt2-chinese-cluecorpussmall 模型，用 [GitHub - gaussic/Chinese-Lyric-Corpus: A Chinese lyric corpus which contains nearly 50,000 lyrics from 500 artists](https://github.com/gaussic/Chinese-Lyric-Corpus) 数据集进行微调，使 gpt2 模型能够生成中文。训练设置和英文类似。

训练：

```python
python finetune_chinese_lyrics.py
```

生成：

```python
python generate_chinese_lyrics.py
```

生成时的输入文本在代码文件中修改。

评估：

```python
python eval_bertScore_chinese.py
```

## 使用 GPT-2 模型制作 chatbot

GPT-2 是一个自回归模型，最基础的功能只能生成文本，不能进行对话。为了使 GPT-2 模型能够对话，我们使用 NaturalConv 多轮中文对话数据集。我们给 BertTokenizer 和 gpt2-chinese-cluecorpussmall 添加了 [speaker1] 和 [speaker2] 两个 token，分别代表 user 和 bot。训练时我们参考([GitHub - thu-coai/CDial-GPT: A Large-scale Chinese Short-Text Conversation Dataset and Chinese pre-training dialog models](https://github.com/thu-coai/CDial-GPT))中的方法，给一次对话中的每条语句的前面轮流添加上述两个 token，并在每句话后面加上 [SEP] 表示说话结束，最后将所有语句拼接，开头加上[CLS]，形成一条数据。

使用时用一个 history 数组保存对话历史。用户每次输入时，将输入语句添加到 history 中，然后使用和训练时一样的方法处理 history 中的语句，形成输入数据，并在最后添加 [speaker2] 以提示模型接下来轮到 bot 说话。得到模型的输出后按 [SEQ] 分割语句并找到模型生成的第一条语句作为模型生成的回答进行输出，并将该回答添加到 history 中。
