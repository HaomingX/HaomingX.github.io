---
title: 'Uncertainty-Confidence'
date: 2023-09-08
permalink: /posts/2023/09/Uncertainty
tags:
  - Uncertainty
  - Confidence
---

# 深度学习的不确定性和校准

在深度学习模型应用中，理解模型预测的不确定性以及对模型进行校准至关重要。这篇博文将深入探讨不确定性的类型、置信度的概念，以及评估和改进模型校准的方法。

## 几个概念

### 1. 不确定性 (Uncertainty) 是什么？

不确定性反映了模型预测结果的可变性和不可靠程度。在深度学习中，我们通常区分以下几种不确定性：

*   **偶然事件不确定性 (Aleatoric Uncertainty)**：
    *   也称为 **数据不确定性** 或 **固有不确定性**。
    *   源于数据生成过程本身的随机性或噪声，例如传感器噪声、图像模糊、自然语言的歧义性等。
    *   这种不确定性即使收集更多数据也无法完全消除，但可以通过**优化数据采集策略**来降低。
    *   **例子**：图像分类中，如果图像本身质量很差（模糊、噪声），模型就很难给出非常确定的预测，即使模型训练得再好，这种图像固有的信息不足导致的不确定性依然存在。

*   **认知不确定性 (Epistemic Uncertainty)**：
    *   也称为 **模型不确定性** 或 **知识不确定性**。
    *   源于模型自身的不完善，例如训练数据不足、模型结构限制、参数估计不准确等。
    *   这种不确定性是**可以**通过收集更多数据、改进模型结构或训练方法来降低的。
    *   **例子**：如果训练数据量不足以覆盖所有可能的输入空间，模型对于未见过的数据区域就会存在不确定性。增加训练数据，特别是多样化的数据，可以帮助模型更好地学习并减少这种不确定性。

*   **超出分布的不确定性 (Out-of-Distribution Uncertainty, OOD)**：
    *   指模型在遇到 **分布外 (OOD)** 数据时产生的不确定性。
    *   模型在训练时学习的是训练数据的分布，当输入数据来自与训练数据分布显著不同的新领域时，模型往往会给出不可靠的预测。
    *   **异常检测** 是处理 OOD 问题的重要方向。
    *   **域内 (In-domain)、域外 (Out-domain) 和开放域 (Open-domain) 的概念：**
        *   **In-domain**:  测试数据与训练数据来自同一分布。例如，使用报纸文本数据训练的模型，用报纸文本数据测试。
        *   **Out-domain**: 测试数据与训练数据来自不同分布。例如，使用报纸文本数据训练的模型，用维基百科文本数据测试。
        *   **Open-domain**: 测试数据分布更广泛，可能包含多种不同的分布。例如，将报纸文本和维基百科文本混合成一个数据集用于测试。

### 2. 置信度 (Confidence) 是什么？

*   **简单定义：** 置信度是对模型决策的信心程度。
*   **模型层面：** 在深度学习模型中，通常使用模型的 **logits (未归一化的输出)** 来反映置信度。更高的 logits 值通常对应更高的置信度。
*   **概率论/机器学习中的置信度与置信区间：**
    *   **置信度**:  对事件估计的信心。
    *   **置信区间**:  在给定的置信度下，估计参数或事件可能落入的范围。
    *   **参数估计**: 利用样本信息估计总体参数。包括：
        *   **点估计**:
            *   **矩估计**: 用样本矩估计总体矩 (基于大数定理)。
            *   **最大似然估计 (MLE)**:  寻找使数据似然性最大化的参数。
        *   **区间估计**: 提供参数估计的范围和置信度。
    *   **假设检验**:  例如 t 检验，用于判断在统计意义上学习器 A 是否优于 B，以及结论的可信程度。

### 3. 概率型模型评估指标

评估概率型模型预测性能，除了常用的准确率、精确率、召回率等指标外，还需要关注模型预测概率的准确性。以下是一些常用的概率型模型评估指标：

1.  **布里尔分数 (Brier Score)**

    *   衡量概率预测与真实标签之间的均方误差。
    *   公式： $$Brier Score=(p−o)^2$$
        *   其中，`p` 是模型预测的概率，`o` 是真实标签 (0 或 1)。
    *   布里尔分数越低，模型预测的概率越准确。

2.  **对数似然函数 (Log Loss) / 二元交叉熵损失 (Binary Cross-Entropy Loss)**

    *   衡量模型预测概率的准确性，值越小表示概率估计越准确。
    *   公式： $$Log Loss=−(y⋅log(p)+(1−y)⋅log(1−p))$$
        *   其中，`y` 是真实标签 (0 或 1)，`p` 是模型预测为正类的概率。

3.  **可靠性曲线 (Reliability Curve) / 校准曲线 (Calibration Curve)**

    *   **图形化工具**，评估分类模型概率预测准确性。
    *   帮助理解模型预测概率与实际观测频率之间的关系，判断模型是否具有良好的概率校准性。
    *   **横轴 (x 轴)**: 模型预测的概率。
    *   **纵轴 (y 轴)**: 实际观测的频率 (在给定预测概率范围内的实际样本比例)。
    *   **理想情况**:  可靠性曲线上的点应落在对角线 y = x 上，表示模型预测的概率与实际频率一致。
    *   **偏离对角线**:  表明模型概率校准性不足，可能需要校准。

    ![可靠性曲线示例](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230829205256146.png)

    *   **逻辑回归的特殊性**: 逻辑回归在设计上具有良好的校准性，因为其目标函数最小化对数损失函数。

4.  **预测概率的直方图**

    *   显示分类模型对不同类别的概率预测的分布情况。
    *   帮助理解模型对于不同类别的预测置信程度分布。
    *   **横轴 (x 轴)**: 概率范围 (0 到 1)。
    *   **纵轴 (y 轴)**: 落在对应概率区间内的样本数量。
    *   **信息**:
        *   **概率分布**:  模型概率预测在不同区间的分布情况。
        *   **置信程度**:  直方图集中程度反映模型置信程度。高度集中表示模型对概率估计比较自信，分散表示不确定性较大。
        *   **校准性**:  校准良好的模型，直方图应大致匹配实际观测的频率分布。

    ![预测概率直方图示例](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230829205530721.png)

### 4. 不确定性估计 (Uncertainty Estimates)

*   计算 **置信度** 和 **真实准确率** 之间的误差程度，得到不确定性的概率值。
*   不确定性 (uncertainty)  同样也反映了置信度，不确定性越高，置信度越低。
*   **目标**:  关注模型对预测结果的把握程度，从而为模型排除错误提供依据。

## Calibration (校准)

### 1. 评价指标

校准的目的是使模型的预测概率更加真实可靠，以下是一些常用的校准评估指标：

1.  **ECE - 预期校准误差 (Expected Calibration Error)**

    *   衡量模型预测概率的平均校准误差。
    *   基于可靠性曲线计算。
    *   将预测概率划分为多个区间 (bins)，计算每个区间内预测概率的平均值与实际准确率的平均值之差的绝对值，并进行加权平均。
    *   公式：

        ![ECE 公式](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230830154848108.png)

        *   其中，`B` 是 bin 的数量，`n_b` 是第 `b` 个 bin 中的样本数量，`acc(b)` 是第 `b` 个 bin 中样本的真实准确率，`conf(b)` 是第 `b` 个 bin 中模型预测概率的平均值。

    *   **Soft ECE**:  一种平滑版本的 ECE，使用连续函数计算权重和平均值，使其可微分，更适合训练过程。
        *   公式：
            $$Soft-Binned ECE = \sum_{i=1}^n \frac{w_i}{\sum_{j=1}^n w_j} |a_i - c_i|$$
            *   `n` 是分区数量，`w_i` 是第 `i` 个分区的权重，`a_i` 是第 `i` 个分区内的预测概率的平均值，`c_i` 是第 `i` 个分区内的真实概率的平均值。
            *   权重计算：$$w_i(x,y)=max(0,1−∣i−k∣)$$, 其中 $$k = ⌊ n \cdot p ⌋+1$$, $$p = f(x)$$ 是预测概率。
            *   平均预测概率贡献：$$a_i(x,y)=w_i(x,y)p$$
            *   平均真实概率贡献：$$c_i(x,y)=w_i(x,y)I[y=argmax(p)]$$，其中 $$I$$ 是指示函数。

    *   **Region ECE**: 将样本划分为不同的区域 (Region) 计算 ECE。

2.  **MCE - 最大校准误差 (Maximum Calibration Error)**

    *   衡量模型在所有区间上的最大校准误差。
    *   公式：

        ![MCE 公式 1](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230718151732832.png)

        ![MCE 公式 2](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230721204209858.png)

    *   MCE 取所有 bin 中 $$|acc(b) - conf(b)|$$ 的最大值。

3.  **负对数似然 (Negative Log Likelihood, NLL)**

    *   衡量模型预测概率与真实标签之间的一致性，越小越好。
    *   公式：

        ![NLL 公式](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230830145609133.png)

4.  **可靠性曲线 (Reliability Diagrams)**

    *   可视化校准情况。理想情况下，曲线应接近对角线。

5.  **AvUC - 平均不确定性校准 (Average Uncertainty Calibration)**

    *   Soft AvUC 是 AvUC 的改进版本，将离散分区变为连续化，使其可微分。
    *   公式：
        $$Soft AvUC = \sum_{i=1}^n \frac{w_i}{\sum_{j=1}^n w_j} |v_i - u_i|$$
        *   其中，$v_i$ 和 $u_i$ 分别是第 i 个分区内的预测概率方差和真实概率方差的平均值。

6.  **AUROC - 接收器操作特征曲线下面积 (Area Under the Receiver Operating Characteristic curve)**

    *   评估置信度分数在 **区分正确和错误样本方面的能力**。
    *   AUROC 越高，模型区分能力越强。

7.  **AUPRC - 精度-召回曲线下面积 (Area Under the Precision-Recall Curve)**

    *   **AUPRC-Positive (PR-P)**:  衡量模型 **识别正确样本** 的能力。
    *   **AUPRC-Negative (PR-N)**:  衡量模型 **识别错误样本** 的能力。

### 2. 校准方法

校准方法可以分为先验校准和后验校准，以及参数化方法和非参数化方法。

*   **先验校准 (Prior Calibration)**：在模型训练过程中或训练前进行校准。
*   **后验校准 (Post Calibration)**：在模型训练完成后，对模型输出的概率进行校准。
*   **参数化方法**: 假设校准映射属于参数化族，通过最小化损失函数学习参数。
*   **非参数化方法**: 不对校准映射做参数假设，通过直方图分箱、贝叶斯平均、保序回归等方法学习。

**2.1 先验校准方法**

1.  **数据增强 (Data Augmentation)**

    *   **MixUp**: 通过线性插值混合样本及其标签，增强模型的泛化能力和校准性。
        *   [MixUp 详解](https://blog.csdn.net/ouyangfushu/article/details/87866579)
    *   **EDA (Easy Data Augmentation)**:  NLP 领域常用的数据增强方法，包括同义词替换、随机插入、随机交换、随机删除等。
        *   [EDA 代码](https://github.com/zhanlaoban/EDA_NLP_for_Chinese/blob/master/code/eda.py)
    *   **标签平滑 (Label Smoothing, LS)**:  软化硬标签，减少模型过度自信，提高 OOD 数据上的性能。
        *   MLE (最大似然估计) 在 In-domain 数据上表现更好，LS 在 Out-of-domain 数据上表现更好。
        *   [标签平滑详解](https://blog.csdn.net/weixin_44441131/article/details/106436808)

2.  **Few-shot Learning**

    *   利用少量样本进行学习，提高模型在数据稀缺情况下的泛化能力和校准性。

3.  **COT (Chain-of-Thought)**

    *   通过引导模型进行逐步推理，提高模型在复杂任务上的性能和置信度。

4.  **Deep Ensemble (深度集成)**

    *   训练多个模型，并将它们的预测结果进行集成，降低模型不确定性，提高鲁棒性和校准性。
    *   **步骤**：
        *   选择合适的神经网络结构和参数，以及集成数量和方式。
        *   对不同的神经网络进行随机初始化和训练。
        *   对它们的输出进行平均或统计 (例如，求平均概率)。
    *   模型集成可以提高模型的复杂度和多样性，从而影响模型的泛化能力和稳定性。

5.  **魔改损失函数 (Modified Loss Functions) / 正则化 (Regularization)**

    *   不良校准通常与负对数似然 (NLL) 的过度拟合有关。通过修改损失函数或添加正则化项，可以改进模型校准。
    *   **Dice Loss**:  常用于图像分割任务，可以缓解类别不平衡问题。
        *   [Dice Loss 代码](https://blog.csdn.net/hqllqh/article/details/112056385)
    *   **LogitNorm 交叉熵损失 (Logit Normalization Cross-Entropy Loss)**:  改进传统交叉熵，解决神经网络过拟合和模型校准问题，适用于知识迁移任务。
        *   [LogitNorm 交叉熵损失论文](https://arxiv.org/abs/2205.09310)
    *   **On/Off-manifold regularization**:  在损失函数上添加正则项，促使模型学习更鲁棒的特征表示。
    *   **Distance-based logits & One vs ALL**:  使用基于距离的 logits 和 One vs ALL 策略改进模型校准。
        *   **Distance Metric (DM) Logits**:  logits $$z_j$$ 定义为嵌入和最后一层权重之间的负欧氏距离：$$zj = − ||f θ(x) − w_j||^2$$

            ![Distance Metric Logits](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230725170352955.png)

        *   **One vs ALL**:  将多分类问题转化为多个二分类问题。

            ![One vs ALL 1](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230725170426142.png)

            ![One vs ALL 2](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230725170446060.png)

        *   **Smooth Softmax**:  在 Softmax 中使用平滑算法。
            *   [Transformer Softmax 平滑算法](https://www.cnblogs.com/liweikuan/p/14253867.html):
                $$W_{i,j} = \frac{\exp(\text{score}(q_i, k_j))}{\sum_{j=1}^{m} \exp(\text{score}(q_i, k_j)) + \exp(\text{extra_logit})}$$

**2.2 后验校准方法**

1.  **Platt Scaling**

    *   使用逻辑回归模型校准分类器的输出概率。
    *   需要将数据集划分为训练集、校准集和测试集。
    *   使用训练集训练原始模型，使用校准集训练 Platt Scaling 模型，使用测试集评估校准后的模型。
    *   **步骤**：
        *   训练原始模型 (例如 AdaBoost)。
        *   使用校准集 (X_calib, y_calib) 和原始模型的预测概率 (clf.predict_proba(X_calib)) 训练逻辑回归校准器。
        *   使用校准器校准测试集 (X_test) 的预测概率 (clf.predict_proba(X_test))。

        ```python
        # uncalibrated model
        clf = AdaBoostClassifier(n_estimators=50)
        y_proba = clf.fit(X_train, y_train).predict_proba(X_test)
        # calibrated model
        calibrator = LogisticRegression()
        calibrator.fit(clf.predict_proba(X_calib), y_calib)

        y_proba_calib = calibrator.predict_proba(y_proba)
        ```

    *   **校准分类器流程**:

        ![Platt Scaling 流程](https://miro.medium.com/v2/resize:fit:700/1*BzjuduA4cxiy3Aoek9r_2A.png)

2.  **Isotonic Regression (等保回归)**

    *   使用保序回归方法校准概率，保证输出的单调性。
    *   将一组无序的数变为有序，通过迭代调整数值，使其保持单调递增或递减。
    *   校准数据集应与训练数据集不同，以避免引入偏差。
    *   **例子**:  将 {1, 3, 2, 2} 变为 {1, 2.5, 2.5, 2.5} 的过程。

3.  **温度缩放 (Temperature Scaling, T scaling)**

    *   对模型的 logits 进行缩放，调整 Softmax 的输出概率。
    *   公式：

        ![温度缩放公式](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9WQmNEMDJqRmhnbndXWmMxZlF6RzJYam9GZTZZRXFpYWxRYUJWaGJpYVJaQTBMQ0dlazZ5VkZ6WDBhNnVQWFJ1MTVNQVo2SXE5ODhqcVBLTWljN0kzUGQyQS82NDA?x-oss-process=image/format,png)

    *   `T` 是温度参数，需要在验证集上优化学习，不能在训练集上学习。
    *   温度缩放是一种后处理步骤，主要用于分类任务。

    *   **改进方法**:
        *   **NRTP (Neural Rank-Preserving Transforms)**:  为每个样本设置不同的温度参数 $$T(x)$$, 而不是全局的 $$T$$。
            *   公式：$$f_{T_θ}(z_b;x) = z_b/T_θ(x)$$
            *   使用单调的两层网络实现，保证排序关系和精确度。
            *   $$g_θ(z_i;x) = \sum_{j} a_jφ((z_i - b^θ_j(x))/T^θ_j(x))$$, 其中 $$a_j \ge 0, T^θ_j(x) > 0, φ$$ 是单调非线性激活函数。
        *   **Region-dependent temperature scaling**:  分区域进行温度缩放，不同区域使用不同的温度参数。

4.  **MC-Dropout (蒙特卡洛 Dropout)**

    *   在 inference 阶段也使用 Dropout，进行多次前向传播，并将结果进行集成，估计模型的不确定性。
    *   **步骤**:
        *   对于一个样本，随机进行 K 次 Dropout。
        *   进行 K 次前向传播，得到 K 个输出结果。
        *   将 K 个输出结果进行 ensemble (例如，求平均概率)。

        ![MC-Dropout 流程](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230914145040644.png)

### 3. 其他论文 (大模型相关)

1.  **[Teaching models to express their uncertainty in words](https://ar5iv.labs.arxiv.org/html/2205.14334)**
    *   **口头表达的置信度**:  用语言文本直接微调模型的 confidence 输出。

2.  **[Large Language Models Are Reasoning Teachers](https://arxiv.org/abs/2212.10071)**
    *   **蒸馏方式**:  使用大模型的输出微调小模型，提高小模型的性能和校准性。

3.  **[Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://ar5iv.labs.arxiv.org/html/2306.13063)**
    *   **LLMs 的不确定性评估**:  使用模型生成的文本输出以及对同一问题的多次回答之间的一致性来衡量模型的不确定性。
    *   **非基于 Logits 的置信度获取**:  探索不需要模型微调或访问专有信息的置信度唤起方法 (confidence elicitation)。
    *   **COT + Consistency**:  结合 COT 和 Consistency 实现更好的校准。

        ![LLM Confidence Calculation](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230921193858057.png)

        *   公式：$$Confidence=\sigma(W_1\cdot LCE + W_2\cdot NLCE + b)$$
            *   LCE (Likelihood Confidence Estimation)
            *   NLCE (Negative Log-Likelihood Confidence Estimation)

4.  **[Making Pre-trained Language Models both Task-solvers and Self-calibrators](https://aclanthology.org/2023.findings-acl.624/)**
    *   **无监督训练置信度**:  提出无监督方法训练 PLMs 的置信度，使其同时具备任务解决能力和自校准能力。
    *   **挑战**:  有限的训练样本、数据不平衡、分布偏移。
    *   **基准方法**:  Vanilla (原始概率)、温度缩放 (TS)、标签平滑 (LS)。

        ![Self-calibrating PLM](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230721140312656.png)

    *   **半监督学习补充**:  PI Model (Perturbation-Invariant Model) 框架。
        *   [半监督学习 PI Model](https://zhuanlan.zhihu.com/p/387907614)

        ![PI Model](https://pic4.zhimg.com/80/v2-1d16af140e9cc3432792ac339db9a85f_720w.webp)

5.  **[Thrust: Adaptively Propels Large Language Models with External Knowledge](https://arxiv.org/pdf/2307.10442v1.pdf)**
    *   **知识增强的 LLM**:  提出 IAPEK 模块 (Instance-level Adaptive Propulsion of External Knowledge)，自适应地检索外部知识。
    *   **Trust 置信度估计模块**:  评估模型对当前输入的信任程度。

        ![Thrust 工作流程](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230721131558174.png)

        ![Trust 置信度估计](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230721135338307.png)

        *   $$s(q)$$ 是 Trust 打分函数，基于查询向量和聚类质心之间的距离计算。

6.  **[Calibration, Entropy Rates, and Memory in Language Models](https://arxiv.org/abs/1906.05664)**
    *   **熵放大现象**:  发现先进语言模型中普遍存在的熵放大现象。
    *   **基于校准的生成改进**:  基于模型长期属性 (熵率) 的测量不匹配来改进生成，并提供可证明的保证。
    *   **量化模型预测与远期过去的依赖关系**:  使用校准方法量化模型预测与远期过去的依赖关系。
    *   **KL 散度补充**:  KL 散度 (Kullback–Leibler divergence) 用于衡量概率分布之间的差异。
        *   离散概率分布 KL 散度公式：
            $$D_{K L}(P \| Q)=\sum_{x \in X} P(x) \ln \left(\frac{P(x)}{Q(x)}\right)=\sum_{x \in X} P(x)(\ln (P(x))-\ln (Q(x))) \text {. }$$
        *   连续概率分布 KL 散度公式：
            $$D_{K L}(P \| Q)=\int_{-\infty}^{+\infty} p(x) \ln \left(\frac{p(x)}{q(x)}\right) d x$$

## 多分类数据不平衡与 Overconfidence

多分类数据不平衡也会导致模型 overconfidence (过度自信)。

**解决数据不平衡的方法：**

*   **数据级方法 (Data-level methods)**:  预处理阶段，与分类器无关。
    1.  **SMOTE (Synthetic Minority Over-sampling Technique)**:  利用 KNN 线性内插生成少样本，同时减少多样本。
    2.  **DeepSMOTE**:  使用 GANs 生成更逼真的合成样本。

*   **算法级方法 (Algorithm-level methods)**:  修改网络和模型的权重、代价敏感学习 (cost-sensitive learning) 等。

*   **Ensemble (模型集成)**:  集成多个模型，提高鲁棒性和泛化能力。

**Loss Function (损失函数改进):**

*   **二进制交叉熵损失 (Binary Cross-Entropy Loss, BCE)**:  常用于多标签文本分类。
    *   公式：$$BCE = -1/N * \sum_{i=1}^N [\sum_{j=1}^C (y_{ij} \cdot log(P(y_{ij})) + (1 - y_{ij}) \cdot log(1 - P(y_{ij})))]$$
        *   `N` 是训练实例数量，`C` 是类别数量，$$y_{ij}$$ 是第 `i` 个实例的第 `j` 个标签 (0 或 1)，$$P(y_{ij})$$ 是模型预测的概率。
    *   纯粹的 BCE 容易受头部类别或负实例支配，导致标签不平衡问题。

*   **平衡 BCE 方法 (Balanced BCE methods)**:  重新加权 BCE，使罕见实例-标签组合获得更多关注。
    *   令 $$p_t$$ 表示模型预测的正确类别的概率，交叉熵损失可写为 $$LOSS_{ce} = -log(p_t)$$。

    1.  **Focal Loss (FL)**:  通过引入焦点因子调整损失函数，更关注难分类样本，减轻类别不平衡问题。
        *   公式：

            ![Focal Loss 公式](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/image-20230906163115279.png)

        *   可以写为 $$ L_{FL} = -(p_t)^γ* log(p_t)$$, 其中 $$γ \ge 0$$ 是焦点参数。
        *   **Inverse Focal Loss (逆焦点损失)**:  与 Focal Loss 相反，增加容易分类样本的权重，减少难分类样本的权重。

    2.  **Class-balanced Focal Loss (CB)**:  类别平衡的 Focal Loss。

    3.  **Distribution-balanced Loss (DB)**:  分布平衡损失。
        *   [DB Loss 详解](https://blog.csdn.net/weixin_42437114/article/details/127774342)

## 一点小思考

1.  **难样本问题**:
    *   难分类样本与易分类样本是一个动态概念，会随着训练过程而变化。
    *   原先易分类样本 ($$p_t$$ 大) 可能变为难训练样本 ($$p_t$$ 小)。
    *   Loss 梯度中，难训练样本起主导作用，参数变化主要朝着优化难训练样本的方向改变。
    *   参数变化可能使原先易训练的样本 $$p_t$$ 发生变化，变为难训练样本，导致模型收敛速度慢。
    *   **解决方法**:  选取小的学习率，防止学习率过大导致 $$w$$ 变化较大，引起 $$p_t$$ 巨大变化，造成难易样本的频繁改变。

2.  **训练大模型时利用负样例和 Loss 函数平滑的思想类似吗？** (待进一步思考和研究)

3.  **如何表示生成式大模型的置信度？**
    *   **方法**:  求和或平均 logits、加权平均、序列模型、注意力机制等。
    *   **目标**:  捕捉序列中不同部分的重要性，提高整体置信度的准确性。

4.  **如何提高大模型的置信度？**
    *   **方法**:  数据增强、模型集成、知识蒸馏、微调等。
    *   **目标**:  提高模型预测的准确性和可靠性。

5.  **如何让大模型认识到自己的知识边界？**
    *   **方向**:  OOD 检测、不确定性估计、知识图谱结合等。
    *   **目标**:  使模型能够判断输入是否在其知识范围内，并给出相应的置信度或拒绝回答。