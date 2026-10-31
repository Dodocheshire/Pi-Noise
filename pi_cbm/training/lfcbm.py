"""Train a fresh CBM using LF-CBM's projection recipe and unmodified SAGA.

The upstream recipe selects the projection on the evaluation split. This is
explicitly configured and reported; adapter validation stays a separate split.
"""
# reviewed
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..models.cbm import ConceptBottleneck
from ..vendor.lfcbm.elasticnet import IndexedTensorDataset, glm_saga
from ..vendor.lfcbm.similarity import cos_similarity_cubed_single


def cos_cubed(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Columnwise teacher agreement used in adaptation diagnostics."""
    # (N,C) 不同列对应不同概念的响应向量
    # 逐元素立方、然后计算C对响应向量的余弦相似度
    target = (target - target.mean(0)).pow(3)
    predicted = (predicted - predicted.mean(0)).pow(3)
    return (F.normalize(target, dim=0) * F.normalize(predicted, dim=0)).sum(0)


def cbm_training_data(data, include_adapter_val):
    """Reassemble official CIFAR training order without duplicating image caches."""
    train = data["splits"]["train"]
    if not include_adapter_val:
        return train
    val = data["splits"]["val"]
    # train["indices"] 和 val["indices"]都是`prepare_data`之前的官方train集而言的
    order = torch.cat([train["indices"], val["indices"]]).argsort()
    return {
        # lfcbm不需要缓存的`internal`data或`images`data
        key: torch.cat([train[key], val[key]])[order] # [order]让"indices"数组是升序的
        for key in ("features", "teacher", "labels", "indices")
    }


def fit_projection(train_x, train_s, val_x, val_s, settings, device):
    """Match upstream minibatch sampling, optimizer and first non-improvement stop."""
    # 投影层无偏置项
    projection = nn.Linear(train_x.shape[1], train_s.shape[1], bias=False).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=settings["projection_lr"])
    indices = list(range(len(train_x)))
    batch_size = min(settings["projection_batch_size"], len(indices))
    best_loss = float("inf")
    history = []
    for step in range(settings["projection_steps"]):
        # 每个 step 独立有放回地从全体训练索引里抽 batch_size 个
        batch = torch.tensor(random.sample(indices, k=batch_size))
        prediction = projection(train_x[batch].to(device))
        # 比较神经元与概念在这个batch上的响应模式的coscube相似度
        loss = -cos_similarity_cubed_single(train_s[batch].to(device), prediction).mean()
        loss.backward()
        optimizer.step()
        # 每50步在验证集上计算1次-coscube平均值作为score，越小越好。并更新一次best_loss，如果比best_loss高，就进行早停
        if step % 50 == 0 or step == settings["projection_steps"] - 1:
            with torch.no_grad():
                score = (
                    -cos_similarity_cubed_single(val_s.to(device), projection(val_x.to(device)))
                    .mean()
                    .item()
                )
            history.append({"step": step, "val_loss": score})
            print(f"Projection step {step}: val_similarity={-score:.6f}", flush=True)
            if score < best_loss:
                best_loss, best_step = score, step
                best_weight = projection.weight.detach().cpu().clone()
            else:
                break
        optimizer.zero_grad()
    # 返回验证集上最佳投影权重
    return best_weight, {"best_step": best_step, "val_similarity": -best_loss, "history": history}


def fit_saga_head(head, train_x, train_y, val_x, val_y, settings):
    """Use the exact glm_saga call in upstream train_cbm.py, including lambda/alpha."""
    train_loader = DataLoader(
        IndexedTensorDataset(train_x, train_y), batch_size=settings["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=settings["batch_size"], shuffle=False
    )
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    output = glm_saga(
        head,
        train_loader,
        settings["lr"],
        settings["steps"],
        settings["alpha"],
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata={"max_reg": {"nongrouped": settings["lambda"]}},
        n_ex=len(train_x),
        n_classes=head.out_features,
    )["path"][0]
    head.load_state_dict({"weight": output["weight"], "bias": output["bias"]})
    return {
        "solver": "upstream_glm_saga",
        "effective_lambda": float(output["lam"]),
        "metrics": output["metrics"],
        "nonzero_gt_1e_5": int((head.weight.abs() > 1e-5).sum()),
    }


def fit_lfcbm(data, config, device):
    settings = config["baseline"]
    train = cbm_training_data(data, settings["include_adapter_val"])
    validation = data["splits"][settings["validation_split"]]
    # Upstream evaluates CIFAR in its original order. The feature cache is
    # shuffled for adapters, so restore the order before projection selection.
    order = validation["indices"].argsort()
    val_x, val_y = validation["features"][order], validation["labels"][order]
    # 每个concept的top-5激活图片的激活值平均
    activation = train["teacher"].topk(5, dim=0).values.mean(0)
    active = activation > settings["activation_cutoff"]
    # 按列（概念）进行过滤
    train_s, val_s = train["teacher"][:, active], validation["teacher"][order][:, active]
    # 传入的train_x是backbone的global features
    weight, projection_metrics = fit_projection(
        train["features"], train_s, val_x, val_s, settings, device
    )
    with torch.no_grad():
        # fit projection后根据cos-cubed值再过滤一次概念
        scores = cos_similarity_cubed_single(
            val_s.to(device), nn.functional.linear(val_x.to(device), weight.to(device))
        ).cpu()
        keep = scores > settings["interpretability_cutoff"]
        selected = active.nonzero().flatten()[keep]
        # 过滤后的概念idx由selected指定
        cbm = ConceptBottleneck(train["features"].shape[1], len(selected), len(data["classes"]))
        # 投影矩阵按行（概念）筛选
        cbm.projection.weight.copy_(weight[keep])
        # Same sample standard deviation and FP32 CPU calculation as upstream.
        raw_train = cbm.projection(train["features"])
        #! 这里projection训练完后，再做一次投影，将统计量保存到buffer `concept_mean` & `concept_std`
        cbm.concept_mean.copy_(raw_train.mean(0))
        cbm.concept_std.copy_(raw_train.std(0))
        # 使用训练集统计量标准化后的train_c & val_c
        train_c = (raw_train - cbm.concept_mean) / cbm.concept_std
        val_c = cbm.concepts(val_x)
    cbm = cbm.to(device)
    head_metrics = fit_saga_head(cbm.head, train_c, train["labels"], val_c, val_y, settings["head"])
    provenance = json.loads(
        (Path(__file__).parents[1] / "vendor/lfcbm/provenance.json").read_text()
    )
    metrics = {
        "method": "lfcbm_from_scratch",
        "upstream_commit": provenance["commit"],
        "train_examples": len(train_c),
        "validation_examples": len(val_c),
        "validation_split": settings["validation_split"],
        "include_adapter_val": settings["include_adapter_val"],
        "active_concepts": int(active.sum()),
        "selected_concepts": len(selected),
        "projection": projection_metrics,
        "head": head_metrics,
    }
    return cbm.eval().requires_grad_(False), selected, metrics
