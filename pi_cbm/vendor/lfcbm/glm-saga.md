glm_saga 优化的就是 elastic_loss（vendor/lfcbm/elasticnet.py:102）在整个训练集上的平均（family='multinomial'、group=False）：

$$\min_{W\in\mathbb{R}^{C\times K},\; b\in\mathbb{R}^{C}} \quad \underbrace{\frac{1}{N}\sum_{i=1}^{N}\mathrm{CE}\!\left(Wc_i+b,\;y_i\right)}_{\text{交叉熵（mean，nats）}} \;+\; \underbrace{\lambda\alpha\,\|W\|_1}_{\text{L1}} \;+\; \underbrace{\tfrac12\lambda(1-\alpha)\,\|W\|_F^2}_{\text{L2}}$$

其中 $c_i\in\mathbb{R}^K$ 是第 $i$ 个样本的冻结概念值（已标准化），$y_i$ 是标签


SAGA 维护一张表：a_table[i] = 样本 i 最近一次被抽到时算出的残差（梯度的一部分）
$$a_i=\mathrm{softmax}(Wc_i+b)-e_{y_i}\in\mathbb{R}^C$$
即
```text
预测概率减去 one-hot 标签 = 交叉熵对 logits 的梯度
```
有了 a_i，W 的梯度可以写成外积：
$$g_{W}=\frac{1}{|B|}\sum_{i\in B}a_i\otimes c_i\in\mathbb{R}^{C\times K}, \qquad b\text{ 的梯度 }=\frac{1}{|B|}\sum_{i\in B}a_i$$

后面的过程涉及`近端梯度法（proximal gradient）`.