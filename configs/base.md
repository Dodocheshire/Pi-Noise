# `configs/base.yaml` 逐字段说明

**`base.yaml` 是全量默认层**。所有其他配置（`cifar10.yaml`、`cifar100.yaml`、`cub.yaml`、`places365.yaml`、`imagenet.yaml`、`*_official.yaml`）都直接或间接 `extends` 它，用来补上自己没写的键。

## 1. `experiment`

| 字段 | 当前值 | 含义 |
|---|---|---|
| `name` | `cifar10_local` | 只用于运行目录名：`<runs>/<时间戳>-<name>/`。不参与缓存哈希，同名字段不会冲突。 |
| `seed` | `0` | 全局种子，`seed_all()` 初始化；同时是 `split_seed: null` 时的数据划分种子、生成器初始化种子、训练 shuffle 种子。评估/重拟合还会用 `seed+1000`、`+2000`、`+3000` 派生固定随机流。 |
| `device` | `cuda` | 计算设备，传给数据预处理、骨干、模型和评估。`pin_memory` 只在 `startswith("cuda")` 时为真。可选 `cuda` / `cpu` / `mps`（`mps` 分支在 [randomness.py](pi_cbm/randomness.py) 里有，但 Linux 上无意义）。 |
| `threads` | `4` | `torch.set_num_threads(threads)`，限制 CPU 线程数。 |

## 2. `paths`

| 字段 | 当前值 | 含义 |
|---|---|---|
| `cache` | `artifacts/cache` | **特征缓存根目录**，存放 `features-<hash>.pt`、`baseline-<hash>.pt`、`channel-scale-<hash>.pt`，以及 `*_internal`/`*_images` 的内存映射 `.bin` 文件。相对路径，按 README 约定在仓库根目录启动。 |
| `runs` | `artifacts/runs` | **运行输出根目录**。每次运行生成子目录，含 `config.yaml`（合并后的完整配置）、`checkpoint.pt`、`metrics.json`、`concepts.txt`、`source/` 源码快照。 |

这两个路径可以按数据集改成 `/dev/shm/...` 或绝对路径（`server` 配置就是这么做的）。

## 3. `data`

| 字段 | 当前值 | 含义 |
|---|---|---|
| `dataset` | `cifar10` | 数据集类型，可选 `cifar10` / `cifar100` / `cub` / `places365` / `imagenet`（[config.py:45](pi_cbm/config.py#L45) 校验）。决定 [data.py](pi_cbm/data.py) 用哪个加载器和划分方式。 |
| `root` | `artifacts/data` | 数据根目录。CIFAR 会自动下载到此处；CUB/Places365/ImageNet 由用户提前准备，不会下载。ImageNet 还要求 `root/train`、`root/adapter_val`、`COMPLETE`、`manifest.jsonl`。 |
| `concepts` | `configs/concepts/cifar10_official.txt` | **概念词表文件**，UTF-8、一行一个概念，行序即概念索引。整个文件内容会参与特征缓存哈希。 |
| `train_per_class` | `100` | 每个类别从“训练集扣除 val 后”的分层抽样数量；**0 表示全用**（[stratified_indices](pi_cbm/data.py#L77)）。对 ImageNet 不同：它是与已有固定划分做硬校验，不重新抽样。 |
| `val_per_class` | `30` | 每个类别从训练集划出的验证数量，用于早停/模型选择/SAGA 验证。**必须 ≥1**，否则 `make_splits` 直接报错。对 ImageNet 同样是对已有 `adapter_val` 的硬校验。 |
| `test_per_class` | `50` | 每类测试样本数；0 表示全部。对 ImageNet 无效——ImageNet 的 test 直接使用 `official_val_root` 全部图片。 |
| `batch_size` | `32` | 特征提取、评估、最终 head 重拟合时 `input_loader` 的批大小。**注意这不是训练批大小**，训练批大小是 `training.batch_size`。 |
| `cache_internal` | `true` | 是否把骨干 `prefix` 输出的内部特征（ResNet 的 layer1–3 输出，即注入点输入）写到 `.bin` 并 mmap。`target_internal` 在 cache 模式下**必须为 true**。磁盘占用约 `N×C×H×W×dtype`，但省掉每轮重复过 prefix。 |
| `cache_images` | `false` | 是否把目标骨干预处理后的图像写到 `.bin`。`target_image` 在 cache 模式下**必须为 true**，否则校验报错。占用很大（`N×3×224×224`）。 |
| `split_seed` | `null` | 数据划分种子；`null` 时回落到 `experiment.seed`（[data.py:23](pi_cbm/data.py#L23)）。固定它可以让换实验种子时**数据划分不变**。ImageNet 使用固定划分，忽略此值。 |
| `num_workers` | `0` | 特征提取/流式输入的 DataLoader 工作进程数。cache 模式下适配器训练读取的是内存/显存张量，`input_loader` 会强制 0（[inputs.py:59](pi_cbm/inputs.py#L59)）。 |
| `version` | `cifar10-official-v1` | 数据版本的人类可读标识，代码不直接读取；但它属于缓存 identity，**改它会让缓存整体失效**，适合在数据内容变化时手动 bump。 |

### 这个 section 里没写、但会被 `.get()` 用到的可选键

| 键 | 默认 | 作用 |
|---|---|---|
| `input_mode` | `cache` | `cache`：从 `.pt/.bin` 读特征；`stream`：只存 `image_paths`，训练时现场解码图片并过骨干。仅 `cub`/`imagenet` 允许 `stream`（[config.py:57](pi_cbm/config.py#L57)）。 |
| `internal_dtype` | `float32` | 内部特征落盘精度；`native` 保留骨干原生 dtype（如 CLIP 的 fp16）。`*_official` 配置里用它节省一半磁盘。 |
| `official_val_root` | 无 | **仅 ImageNet**：官方验证集目录，作为 test split，并用 WNID 校验与训练集类别一致。 |


## 4. `backbone`

| 字段 | 当前值 | 含义 |
|---|---|---|
| `name` | `resnet18` | 目标骨干，可选 `resnet18`、`resnet50`、`resnet18_cub`、`clip_RN50`（[backbones.py:120](pi_cbm/models/backbones.py#L120)）。决定：网络权重、预处理、`feature_dim`、内部通道数，以及注入点（`pre_layer4` / `pre_stage4` / `pre_attnpool`）。 |
| `precision`（可选） | 未写 → `float32` | 仅 `clip_RN50` 有效：`native` 保留 CLIP 原生 fp16 计算，`float32` 全程 fp32。`*_official` 配置用 `native`。 |

## 5. `teacher`

| 字段 | 当前值 | 含义 |
|---|---|---|
| `name` | `ViT-B/32` | CLIP 模型，用于两件事：① 把概念词编码成文本 embedding；② 对图像编码得到 teacher target（`visual @ text.T`）。默认 `ViT-B/32`，官方复现用 `ViT-B/16`。 |
| `precision`（可选） | 未写 → `float32` | `native` 时教师用原生精度并在 GPU 上算，否则转 fp32。 |

## 6. `baseline` — 干净 CBM（LF-CBM 概念层）的拟合

这一整块**只影响基线 CBM 的构建**，和后面的噪声适配器无关。若 `import_dir` 非空，下面所有拟合超参都会被跳过。

| 字段 | 当前值 | 含义 |
|---|---|---|
| `import_dir` | `null` | `null`：从零拟合；路径：导入已发布的 LF-CBM checkpoint（CUB 用），直接得到 projection/head/概念选择。 |
| `include_adapter_val` | `false` | 是否把 `val` split 合并回基线训练集。官方 CIFAR 复现设为 `true`，以还原完整 50k 训练集；适配器自己的 val 不受影响。 |
| `validation_split` | `val` | 基线拟合时用于投影早停和 SAGA 验证的 split。官方配置用 `test` 以对齐上游训练脚本（属于上游协议，会在 metrics 里标注）。 |
| `seed` | `null` | 基线拟合种子；`null` 时用 `experiment.seed`。它是基线缓存 key 的一部分。 |
| `activation_cutoff` | `0.20` | **活跃概念过滤**：对每个概念，取训练集 teacher 激活的 top-5 均值，超过该阈值才进入投影训练（[lfcbm.py:115](pi_cbm/training/lfcbm.py#L115)）。 |
| `interpretability_cutoff` | `0.25` | **可解释性过滤**：投影训练后，在验证集上算每列 `cos^3` 相似度，超过该阈值的概念才保留为最终 CBM 概念。 |
| `projection_steps` | `300` | 投影层 Adam 的最大步数；每 50 步验证一次，验证损失不再下降即提前停止。 |
| `projection_lr` | `0.001` | 投影层 Adam 学习率。 |
| `projection_batch_size` | `50000` | 每步随机采样的样本数（实际取 `min(该值, 训练集大小)`）。 |
| `head.steps` | `400` | SAGA 稀疏线性头最大迭代步数。 |
| `head.batch_size` | `256` | SAGA 求解器的 mini-batch 大小。 |
| `head.lr` | `0.1` | SAGA 学习率。 |
| `head.lambda` | `0.001` | L1 正则强度（`max_reg.nongrouped`），控制头权重稀疏度，直接决定 `nec`/`head_nonzero` 指标。 |
| `head.alpha` | `0.99` | Elastic-net 混合系数，1 附近基本等价于纯 Lasso。 |

基线本身也有缓存：`baseline-<hash>.pt`，key 包含整个 `baseline` 块、种子、特征缓存文件名，以及 `lfcbm.py` 和 vendor 代码的哈希（[pipeline.py:59-73](pi_cbm/pipeline.py#L59-L73)）。

## 7. `injection`

| 字段 | 当前值 | 含义 |
|---|---|---|
| `site` | `target_internal` | 噪声注入位置，四选一：<br>• `baseline`：不注入，纯干净 CBM<br>• `target_global`：在池化后的特征向量上注入<br>• `target_internal`：在骨干中间层特征（layer3 输出 / stage3 输出 / attnpool 前）上注入<br>• `target_image`：在输入图像上注入<br>对应类与缓存键见 [sites.py:78](pi_cbm/injections/sites.py#L78)：`baseline→features`、`global→features`、`internal→internal`、`image→images`。 |
| `point` | `pre_layer4` | **声明式校验字段**，运行时不读取。它只用于 [config.py:67-81](pi_cbm/config.py#L67-L81) 检查“骨干与注入点是否匹配”：`resnet18/50→pre_layer4`、`resnet18_cub→pre_stage4`、`clip_RN50→pre_attnpool`、`target_image→input`。真正切分点由骨干类自己决定。 |

关联约束：`target_internal` 需要 `cache_internal=true` 或 `input_mode=stream`；`target_image` 需要 `cache_images=true` 或 `stream`，且 `point=input`。

## 8. `noise` — 噪声生成器

| 字段 | 当前值 | 含义 |
|---|---|---|
| `generator` | `concept_cross_attention` | 生成器结构，可选 `mlp` / `concept_cross_attention`。后者以概念文本 embedding 为 key/value 做注意力，输出 `query + attn·value`；前者只是 `Linear+GELU`。**`parameterization=fixed_gaussian` 时此字段被忽略**。 |
| `parameterization` | `mean_scale` | `mean_scale`：学习对角高斯 `μ(x), σ(x)`；`fixed_gaussian`：无参数，`μ=0`、`σ=initial_scale×通道RMS`（[noise.py:54-77](pi_cbm/generators/noise.py#L54-L77)）。 |
| `hidden_dim` | `64` | 生成器隐层宽度（MLP 的 hidden / 注意力的投影维度）。 |
| `initial_scale` | `0.02` | 初始噪声尺度，单位是“通道 RMS 的倍数”。`mean_scale` 下通过 `softplus` 偏置初始化成该值；`fixed_gaussian` 下就是固定 σ 系数。必须 >0。 |
| `image_grid` | `14` | **仅 `target_image`**：先把图像池化到 `14×14`，在网格上预测 `μ/σ`，再双线性上采样到原图分辨率后逐像素采样（[sites.py:52-70](pi_cbm/injections/sites.py#L52-L70)）。越小越平滑、参数越少。 |

所有模式都会额外返回一个 `energy = mean((μ/s)² + (σ/s)²)`（`s` 为通道 RMS），供训练里的能量惩罚使用。

## 9. `training` — 适配阶段

| 字段 | 当前值 | 含义 |
|---|---|---|
| `stage` | `generator_only` | 训练阶段：<br>• `generator_only`：只训生成器，最后直接用带噪模型评估<br>• `refit_head`：先训生成器（若存在），然后冻结一切、重新采样一次概念、用 SAGA 重拟合分类头<br>• `finetune_projection`：生成器 + 投影层一起训，并额外加教师对齐损失；最后同样重拟合头<br>注意：只要 `site != baseline`，**三个阶段都会训练生成器**——stage 控制的是“投影是否解冻”和“是否重拟合头”。`baseline` 与 `finetune_projection` 的组合被校验禁止。 |
| `epochs` | `5` | 适配训练轮数；每轮用 `val` 的 NLL 选最佳 checkpoint。 |
| `lr` | `0.001` | 生成器参数的学习率（Adam 默认 β、无 weight decay）。 |
| `projection_lr` | `0.00001` | 投影层学习率，仅 `finetune_projection` 生效。 |
| `grad_clip_norm` | `1.0` | 生成器 + 可训练投影**共享**的梯度范数裁剪阈值。 |
| `batch_size` | `64` | 适配训练批大小（与 `data.batch_size` 独立）。 |
| `concept_anchor_weight` | `1.0` | 干净概念锚定损失权重：`MSE(noisy_concepts, clean_concepts)`，防止噪声把概念表示推太远。 |
| `noise_energy_weight` | `0.001` | 能量惩罚权重：`weight × energy`，抑制过大的均值/方差。 |
| `teacher_alignment_weight` | `1.0` | 教师对齐权重，仅 `finetune_projection`：`weight × (1 - mean cos³(teacher, concepts))`。 |

## 10. 这份 base.yaml 展开后的实际配置

- 数据：CIFAR-10，每类 train/val/test = 100/30/50，共 1000/300/500 张；划分由种子 0 决定。
- 概念：142 个官方概念；CLIP ViT-B/32 提供文本和图像教师目标。
- 基线：ResNet-18 特征 → 投影到概念（20% 活跃阈值、25% 可解释性阈值）→ SAGA 稀疏头。
- 适配：在 ResNet-18 的 `layer3` 输出上注入，生成器用概念交叉注意力预测对角高斯，初始化 σ=0.02×通道 RMS，只训 5 个 epoch。
- 设备：CUDA，4 线程；缓存和运行结果都在 `artifacts/` 下。

它本身是一个合法可跑的配置（小规模），但 [scripts/train.py](scripts/train.py) 和 [scripts/prepare.py](scripts/prepare.py) 的默认入口已经改成 `configs/cifar10.yaml`（每类 500 val、20 epoch 的服务器基线）。

---

## 注意 

1. **`injection.point` 目前只是校验元数据**，改它不会改变实际注入位置；实际位置由 `site` + `backbone.name` 决定。如果以后支持同一骨干多个切分点，才需要让它进入运行时。