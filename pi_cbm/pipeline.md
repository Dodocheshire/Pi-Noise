以 target_internal + cache 为例：

```python
input = split["internal"]          (B,1024,14,14)   ← layer3 输出
  prepare_inputs: prefix=None  → 原样返回
  InternalInjection.forward:
    tokens, energy = generator(to_tokens(input))    (B,196,1024)
    from_tokens(...)                                (B,1024,14,14)
  finish: suffix(perturbed) → layer4+avgpool+flatten (B,512)
          cbm.concepts(...)                          (B,K)
head → logits                                          (B,C)
```
以 target_image 为例：
```python

input = split["images"]            (B,3,224,224)     ← 已按骨干预处理
  ImageInjection.forward: 14×14 网格上预测 μ,σ → 上采样 → 逐像素采样
  finish: suffix(noisy)            ← suffix 就是整条 backbone
          cbm.concepts(...)
```

以 target_internal + stream 为例，多了一步 prepare_inputs：
```python
input = 原图 (B,3,224,224)
  prepare_inputs: prefix(input) 在 no_grad 下算到 layer3 输出
  ...同上
```