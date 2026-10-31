"""Single-sample prediction and diagnostics after the concept bottleneck."""

import torch
from torch.nn import functional as F

from ..inputs import input_loader
from ..training.lfcbm import cos_cubed


def predict(model, inputs, intervention=None):
    """Apply an optional standardized-concept intervention after one noise draw."""
    concepts, _ = model.encode_concepts(inputs)
    if intervention is not None:
        indices, values = intervention
        concepts[:, indices] = values
    return model.cbm.head(concepts).softmax(-1), concepts


@torch.no_grad()
def evaluate(model, split: dict, clean_concepts: torch.Tensor, selected, config: dict, device: str):
    model.eval()
    probability_batches, concept_batches = [], []
    loader = input_loader(split, model, config, batch_size=config["data"]["batch_size"])
    for (batch,) in loader:
        inputs = model.prepare_inputs(batch.float().to(device))
        probabilities, concepts = predict(model, inputs)
        probability_batches.append(probabilities.cpu())
        concept_batches.append(concepts.cpu())
    probabilities, concepts = torch.cat(probability_batches), torch.cat(concept_batches)
    labels = split["labels"]
    weights = model.cbm.head.weight
    return {
        "accuracy": (probabilities.argmax(1) == labels).float().mean().item(),
        "nll": F.nll_loss(probabilities.clamp_min(1e-12).log(), labels).item(),
        "concept_drift_mse": F.mse_loss(concepts, clean_concepts).item(),
        "teacher_cos_cubed": cos_cubed(split["teacher"][:, selected], concepts).mean().item(),
        # VLG-CBM Eq. (9): average effective concepts per class. Its released
        # evaluator treats weights with magnitude <= 1e-5 as numerical zero.
        "nec": (weights.abs() > 1e-5).sum().item() / weights.shape[0],
        "head_nonzero": int(weights.count_nonzero()),
        "head_total": weights.numel(),
        "examples": len(labels),
    }
