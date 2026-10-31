# Pi-CBM实验

## 1. pipeline

在[LF-CBM](https://arxiv.org/abs/2304.06129)的“图像特征→可命名概念→稀疏分类头”路径中，借鉴[PiNI](https://arxiv.org/abs/2412.10817)的可学习高斯扰动，选择不同位置注入噪声，观察分类性能、概念保持程度。

| 环节 | 论文中组件 | 当前实现 | 代码 |
| --- | --- | --- | --- |
| 概念与文本目标 | LF-CBM先建立、筛选概念集合，再用CLIP文本编码器提供概念表示 | CIFAR-10/100使用固定概念列表和冻结的CLIP ViT-B/16文本表示；CUB导入已训练的LF-CBM概念列表及参数 | `configs/concepts/`、`pi_cbm/data.py`、`pi_cbm/models/import_lfcbm.py` |
| CBM基线 | LF-CBM训练图像特征到概念的投影，并使用稀疏线性分类头 | CIFAR-10/100从头训练CLIP RN50特征上的概念投影，按官方激活与可解释性阈值筛选概念，再使用官方`glm_saga`拟合分类头 | `pi_cbm/training/lfcbm.py`、`pi_cbm/models/cbm.py`、`pi_cbm/vendor/lfcbm/` |
| 噪声生成 | PiNI以视觉特征为条件预测对角高斯分布的均值和尺度，再进行重参数采样 | 实现`NoiseGenerator`，默认同时预测均值与标准差；概念条件交叉注意力读取当前图像特征和固定的概念文本向量。概念文本向量本身冻结，注意力中的Query、Key、Value投影可训练| `pi_cbm/generators/noise.py` |
| 注入位置 | PiNI可在原始图像或视觉编码路径的不同位置加入扰动 | 一套注入接口支持`target_image`、`target_internal`、`target_global`：分别作用于预处理图像、骨干中间特征、骨干输出的全局特征。随后沿原骨干/CBM路径计算概念和分类结果。 | `pi_cbm/injections/sites.py`、`pi_cbm/models/backbones.py` |
| 适配训练 | PiNI通过下游监督信号训练噪声；LF-CBM用投影和稀疏头维持概念分类路径 | 统一采用`finetune_projection`：训练生成器和概念投影，损失包含分类交叉熵、与干净概念的均方误差约束、噪声能量约束，以及与CLIP概念目标的`cos_cubed`对齐；按适配验证集NLL选模，再用官方`glm_saga`重拟合稀疏分类头。CIFAR运行还采用梯度范数裁剪(之前有梯度不稳定现象) | `pi_cbm/training/adapt.py`、`pi_cbm/training/lfcbm.py` |
| 评估与复现 | [VLG-CBM](https://arxiv.org/abs/2408.01432)第4.2节提出NEC衡量分类头每类使用的有效概念数 | 统计准确率、NLL、标准化概念漂移、与教师概念的相似度及NEC | `pi_cbm/evaluation/inference.py`、`pi_cbm/pipeline.py`、`reports/`中的JSON/YAML |

计算顺序：
1. 先建立或导入干净CBM，再缓存图像、骨干特征和CLIP概念目标
2. 按配置在三个位置之一生成并注入噪声、训练生成器与投影。冻结这两部分后为每张训练图采样一次概念向量。
3. 以固定的概念表重拟合分类头
4. 在官方测试集上评估。

默认扰动形式为$\epsilon=\mu_\theta(s,T)+\sigma_\theta(s,T)\odot\xi$，其中$s$是待注入特征，$T$是固定概念文本向量集合，$\xi\sim\mathcal N(0,I)$。`target_image`先在**$14\times14$网格**预测高斯参数，再插值到输入图像尺寸，并在图像分辨率逐像素采样。



## 2. 实验结果

### 2.1 CIFAR-10与CIFAR-100

两套CIFAR实验均以CLIP RN50为backbone、CLIP ViT-B/16提供概念文本及教师目标。各用完整50,000张官方训练图从头拟合干净CBM；噪声适配另使用其中45,000张训练、5,000张验证；评估使用10,000张官方测试图。CIFAR-10最终保留132个概念，CIFAR-100保留824个。三种位置均使用概念条件交叉注意力、20轮训练、批量256、生成器学习率$3\times10^{-4}$、投影学习率$10^{-5}$、梯度范数上限1.0。

干净CBM采用**顺序拟合**：先依据CLIP概念目标筛选候选概念，训练图像特征到概念值的投影；再筛选和标准化概念；最后固定投影，用`glm_saga`单独拟合稀疏分类头。投影与分类头没有联合更新。后续`finetune_projection`实验则同时更新噪声生成器和概念投影，完成后才重拟合分类头。

| 数据集 | 注入位置 | 测试准确率 | NLL↓ | NEC↓ | CLIP概念拟合`cos_cubed`↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 干净CBM | 86.46% | 0.3886 | 37.00 | 0.6153 |
| CIFAR-10 | `target_global` | 88.27% | 0.3271 | 21.30 | 0.6132 |
| CIFAR-10 | `target_internal` | **89.88%** | **0.2958** | **18.60** | **0.6298** |
| CIFAR-10 | `target_image` | 87.81% | 0.3440 | 26.10 | 0.6190 |
| CIFAR-100 | 干净CBM | 65.06% | 1.2433 | 27.86 | 0.6076 |
| CIFAR-100 | `target_global` | 69.00% | 1.0633 | **13.92** | 0.5977 |
| CIFAR-100 | `target_internal` | **70.62%** | **1.0061** | 14.00 | 0.6076 |
| CIFAR-100 | `target_image` | 67.26% | 1.1304 | 16.79 | **0.6082** |

数据划分说明：按当前LF-CBM训练协议，干净CBM使用官方测试集选择投影状态和筛选概念，表中又在该测试集报告结果；另外，5,000张适配验证图也参与过第一阶段的干净CBM拟合。因此，这些结果用于当前协议下的模型比较，不能当作完全独立的盲测成绩。

### 2.2 CUB-200-2011

CUB使用导入的LF-CBM ResNet-18检查点作为起点(已用CUB预训练过），保留208个概念。
原训练集5,990张中，4,990张用于噪声适配训练、1,000张用于适配选模，完整官方测试集5,790张用于评估。
三种位置均用概念条件交叉注意力、20轮`finetune_projection`及其后的5,000步官方`glm_saga`分类头重拟合；生成器学习率$10^{-3}$.

| 注入位置 | 正确数/5790 | 测试准确率 | NLL↓ | NEC↓ | CLIP概念拟合`cos_cubed`↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 干净CBM | 4318 | 74.5769% | 1.197203 | 31.520 | 0.5678 |
| `target_global` | 4321 | 74.6287% | 1.194907 | **22.935** | 0.5702 |
| `target_internal` | 4299 | 74.2487% | **1.181536** | 25.815 | **0.5837** |
| `target_image` | 4327 | **74.7323%** | 1.191085 | 25.610 | 0.5680 |

CUB的干净检查点已使用过原训练集，其中包含本次划出的1,000张适配验证图；该验证集只用于当前适配阶段选模，不代表对干净CBM的独立泛化验证。



NEC为：对最终分类头权重$W\in\mathbb R^{C\times K}$，以$|W_{ij}|>10^{-5}$视为一个有效“类别—概念”连接，按类别取平均

$$
\mathrm{NEC}(W)=\frac{1}{C}\sum_{i=1}^{C}\sum_{j=1}^{K}\mathbf{1}\{|W_{ij}|>10^{-5}\}
$$

**NLL（负对数似然）**：所有测试图损失的平均值

**CLIP概念拟合`cos_cubed`**：如果没有逐图的真实概念标注，可以用冻结的CLIP图像编码器与概念文本编码器计算相似度，作为概念层的代理监督信号。对每个概念，比较它在全部测试图上的CLIP相似度与CBM概念输出的变化趋势：两列数值分别减去均值、取三次方，再计算余弦相似度，越高表示越贴近CLIP监督信号。

## 3. 下一步消融、探索与数据集

| 实验 | 目的 |
| --- | --- |
| 加入固定高斯噪声对照。 | 区分随机扰动与可学习扰动的作用。 |
| 保持注入位置不变，将`concept_cross_attention`换成`mlp`，再做打乱概念文本向量的对照。 | 概念文本条件是否真的为噪声生成提供信息|
| 将“生成器读取的特征”与“接收噪声的特征”拆成两个独立配置，例如全局特征预测中间特征噪声；分别处理两侧的形状与通道尺度。 | 探索噪声生成器的输入和注入解耦合的影响 |

- **Places365**：检验加噪声对场景识别能力是否有提升
- **DTD纹理**与**Oxford-IIIT Pets细粒度动物**：检验其他语义数据集的可扩展性
- **ImageNet**：磁盘空间不够了

### Reference

- [Label-Free Concept Bottleneck Models](https://arxiv.org/abs/2304.06129)：概念投影、概念筛选与稀疏分类头的基线框架
- [Enhance Vision-Language Alignment with Noise](https://arxiv.org/abs/2412.10817)：PiNI的可学习高斯噪声、视觉/文本注入与生成器条件设计。
- [Positive-incentive Noise](https://arxiv.org/abs/2212.09541)及[Variational Positive-incentive Noise](https://arxiv.org/abs/2306.07651)：Pi噪声概念与变分建模背景。
- [VLG-CBM](https://arxiv.org/abs/2408.01432)：NEC定义与概念使用量评估。
