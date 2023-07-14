# Goat-Math-Chinese

山羊中文算术大模型

## 介绍

[《Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks》](https://arxiv.org/pdf/2305.14201.pdf)

上述论文提出了一个专供算术的模型山羊Goat，是在LLaMA上进行的微调，由于LLaMA不支持中文，所以我们训练一个基于中文的算术模型。

**中文基座模型**

- [baichuan-7B](https://github.com/baichuan-inc/baichuan-7B)
- [baichuan-13B-chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)
- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)

由于LLaMA不支持中文，所以我们选择一个支持中文的LLaMA模型，最近百川智能开源了baichuan-7B的模型，这个模型和LLaMA是一样的模型设计

在Goat的论文中，对比了不同模型的tokenizer，ChatGLM-6B和LLaMA一样，对数字的每一位进行单独切分，避免了数字不一致的问题，所以我们也来测试一下ChatGLM-6B

**论文原理介绍**

[构建中文小学数学垂类大模型-原理介绍](https://zhuanlan.zhihu.com/p/637999512)

[构建中文小学数学垂类大模型-ChatGLM实战](https://zhuanlan.zhihu.com/p/643492290)


## 数据集

- [goat-chinese](https://huggingface.co/datasets/supinyu/goat-chinese)
- [belle-math-0.25m](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)

Goat论文，开源了英文的Goat数据集，主要包括加减乘除的相关数据，我将其转成成了中文的算术数据集，放到了huggingface上面

Belle也开源了一个数学的数据集，我们也拿来一起训练一下，让模型也能够解答简单的数学问题

模型训练集输入格式

```
instruction: 指令
input: 输入（本数据集均为空）
output: 输出
```
