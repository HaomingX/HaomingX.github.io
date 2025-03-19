---
title: molecular
author: haomingx
avatar: /images/favicon.png
authorDesc: 不断折腾
comments: true
date: 2024-03-18 08:51:12
authorLink:
authorAbout:
categories:
series:
tags:
keywords:
description:
photos:
---

## Paper1: A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals

source: nature sci一区论文

author: 清华刘志远团队 博士生曾哲妮、姚远

论文主要关注分子结构和生物文本信息的统一深度学习框架

**SMILES+BERT+双tokenizer**

分子用SMILES表示，分子和文本使用不同的tokenizer，tokenization之后将分子insert进文本中，然后做"完形填空"预训练，**跨信息检索微调**(这是一个双向检索任务，模型需要为化学描述和SMILES字符串找到最佳的匹配对。这个过程通过计算文本表示的余弦相似度来获得匹配得分，并使用准确率作为评估指标)

![image-20240317001701185](images/image-20240317001701185.png)

## Paper2 Translation between Molecules and Natural Language

UIUC、X以及纽大合作论文

提出MolT5—一种自监督学习框架，用于在大量未标记的自然语言文本和分子字符串上预训练模型。

![image-20240317174327789](images/image-20240317174327789.png)

**预训练方法：** 论文采用了创新的“完形填空”式预训练方法，在预训练阶段处理大量的未标记数据

**微调过程：**分子生成任务**：在这一任务中，预训练模型接收分子的描述性文本作为输入，并输出与描述相匹配的目标分子的SMILES表示。这个过程模拟了从自然语言描述中生成化学结构的过程。**分子标题生成任务：在此任务中，模型的输入是分子的SMILES字符串，而输出则是一个准确描述该分子的文本标题。这要求模型不仅要理解化学结构，还要能够用自然语言准确地表达这些信息。

**评估指标：**

1. **text2mol metric**：这个新指标用于评估模型在将文本描述转换为分子结构时的性能。它通过计算真实分子与模型生成的分子之间的余弦相似度来衡量相似性。该指标利用了分子和文本的embeddings，为评估提供了一种基于内容的量化方法。
2. **分子标题评估**：尽管传统的评估指标如BLEU、ROUGE和METEOR在某些情况下仍然有用，但作者指出它们在当前的应用中可能不够有效。这意味着在评估分子标题生成任务时，可能需要更专门化的指标来准确衡量模型的性能。
3. **基于文本的新分子生成评估**：为了评估从文本生成的分子与输入文本的匹配度，论文采用MACCS FTS、RDK FTS和Morgan FTS三个指纹指标。此外，还考虑了SMILES字符串的匹配度、Levenshtein距离和SMILES BLEU分数。对于基于SMILES字符串的模型，还特别关注了生成分子的有效性（validity），即计算能够被RDKit处理的分子所占的百分比。



## Paper3 Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing

Shengchao Liu(Mila 蒙特利尔学习算法研究所) 等

本文首先指出paper3那种有label的数据形式需要**昂贵**的标注，所以它还是采用了无监督预训练

作者提出了MoleculeSTM，并构建了一个大型多模态数据集PubChemSTM，包含超过280,000对化学结构-文本对（）

![image-20240317172802143](images/image-20240317172802143.png)

模型架构如（a）,是将分子模态预训练模型和文本模态预训练模型通过对比学习映射到同一表示空间在数据集上对齐（使用GraphMVP预训练的GIN模型和MegaMolBART作为化学结构分支的预训练模型，使用SciBERT作为文本描述分支的预训练模型）

文章设计了三个下游任务分别如b,c,d



## Paper4: Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language

微软的Philipp Seidl等

CLAMP(Contrastive Language-Assay-Molecule Pre-training)

一个molecule embedding 一个text encoder 一个评分函数$k(m,a) =  \frac{\exp \left(\tau^{-1} \boldsymbol{m}^{T} \boldsymbol{a}\right)}{\exp \left(\tau^{-1} \boldsymbol{m}^{T} \boldsymbol{a}\right)+1}$(如果分子m对生物测定a有活性，则返回高值，否则返回低值)

模型架构：  $y  = k(f(m), g(a))$

损失函数: (Noise Contrastive Estimation，就是交叉熵的形式)

$\begin{aligned}
\mathrm{L}_{\mathrm{NCE}}=-\frac{1}{N} \sum_{n=1}^{N} y_{n} \log \left(k\left(\boldsymbol{f}_{\boldsymbol{w}}\left(m_{n}\right), \boldsymbol{g}_{\boldsymbol{v}}\left(a_{n}\right)\right)\right)+ 
\left(1-y_{n}\right) \log \left(1-k\left(\boldsymbol{f}_{\boldsymbol{w}}\left(m_{n}\right), \boldsymbol{g}_{\boldsymbol{v}}\left(a_{n}\right)\right)\right.
\end{aligned}$

对比损失函数鼓励**在生物测定中具有活性的分子具有与给定生物测定的嵌入相似的嵌入**，而**非活性分子应具有与之不同的嵌入**。

![image-20240317005932845](images/image-20240317005932845.png)

架构中molecule encoder通过实验了graph, smiles, 以及分子指纹表示分子，作者称分子指纹的方式最好。

## Paper5: MolXPT: Wrapping Molecules with Text for Generative Pre-training

粗暴的方式：

1. **命名实体识别（NER）和实体链接**：首先，使用生物医学领域专用的NER工具（如BERN2）来检测文本中的分子名称，并将这些名称与公共知识库（如ChEBI）中的实体进行链接。
2. **替换分子名称**：链接成功后，将文本中的分子名称替换为对应的SMILES表示。这样，就得到了所谓的“包裹”序列，即SMILES与文本的结合体。
3. **预训练**：将这些**“包裹”序列**与**纯文本序列**（来自PubMed的科学文献标题和摘要）和**纯SMILES序列**（来自PubChem的分子数据）一起输入到语言模型（350M参数的GPT2架构的模型）中进行预训练。预训练的目标是最大化模型对序列的生成概率，即最小化负对数似然。
4. **微调**：预训练完成后，MolXPT可以通过基于提示的微调方法适应各种下游任务，如分子属性预测和文本-分子翻译。

![image-20240317001741465](images/image-20240317001741465.png)

## Paper6 GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning

作者：Haiteng Zhao  (北大)  Shengchao Liu (Mila 蒙特利尔学习算法研究所) Chang Ma(香港大学) Hannan Xu (University of Oxford) 

Jie Fu (香港科技大学)

![image-20240317012053662](images/image-20240317012053662.png)

受自回归大模型的发展影响，作者考虑训练一个指令遵循的分子文本语言模型，不再通过有标签数据的监督微调完成任务而是通过基于指令的zero-shot

作者觉得无论是SMILES将分子表示成序列表示还是GNN把分子图编码成密集向量，都不利于语言模型对图特征深入学习，作者提GIMLET(Graph Instruction based MolecuLe zEro-shoT learning)。

GIMLET模型通过以下方式解决现有问题：

1. 采用Transformer和Generalized position embedding(广义位置嵌入)，将语言模型扩展到图数据和文本数据。
2. 通过注意力掩码解耦图的编码，增强了图特征在新任务中的泛化能力。
3. 构建了一个包含2000多个分子任务的数据集，并进行预训练，使模型能够理解自然语言中表达的特定指令，并将知识转移到广泛的任务中。

本文主要就是通过广义位置嵌入把分子和文本一起嵌入输入了Transformer:

![image-20240317220935000](images/image-20240317220935000.png)

在attention中添加偏置项b(i,j)表示i,j的相对距离

![image-20240317221059597](images/image-20240317221059597.png)

其由三个部分组成： 

1. **Position bias (位置偏置)**：这是位置嵌入的基础，它为图中的节点和文本中的标记提供了一个相对位置的表示。在GIMLET模型中，位置偏置不仅包括序列中元素之间的相对位置（如i - j），还包括图的最短距离（GRAPH SHORTEST DISTANCE），以及用于区分图和文本标记之间交叉距离的特殊标记（<CROSS>）。

   ![image-20240317221204116](images/image-20240317221204116.png)

2. **Masking (掩码)**：为了将图编码与文本指令分离，GIMLET使用了一个称为bM的掩码机制。这个掩码确保图标记只能关注其他图标记，而文本标记则可以接收来自图和文本的信息。这种单向约束有助于模型在执行不同下游任务时，选择性地利用图特征。

   ![image-20240317221409363](images/image-20240317221409363.png)

3. **Path embedding (路径嵌入)**：这是通过考虑图中节点之间的最短路径来增强位置嵌入的一种方法。具体来说，对于图中任意两个节点i和j之间的最短路径SP(i, j)，路径嵌入是路径上边特征bE的均值池化（Mean pooling）。



分子方面总结：

分子表征大概有六个模态：

![image-20240317212258439](images/image-20240317212258439.png)