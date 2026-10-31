"""Train noise, optionally projection, then optionally refit a sparse head."""

from copy import deepcopy

import torch
from torch.nn import functional as F

from ..evaluation import evaluate, predict
from ..inputs import input_loader
from ..randomness import seeded
from .lfcbm import cos_cubed, fit_saga_head


def adapt(model, data, clean_concepts, selected, config, device):
    settings = config["training"]
    model.cbm.requires_grad_(False)
    groups = []
    if model.generator is not None:
        parameters = list(model.generator.parameters())
        if parameters:
            groups.append({"params": parameters, "lr": settings["lr"]})
    if settings["stage"] == "finetune_projection":
        model.cbm.projection.requires_grad_(True)
        groups.append(
            {"params": model.cbm.projection.parameters(), "lr": settings["projection_lr"]}
        )
    history = []
    if groups:
        optimizer = torch.optim.Adam(groups)
        trainable = [parameter for group in groups for parameter in group["params"]]
        train = data["splits"]["train"]
        # The same seed gives the same sample order across noise variants;
        # sampling noise cannot advance the data-shuffle RNG.
        shuffle_rng = torch.Generator().manual_seed(config["experiment"]["seed"])
        loader = input_loader(
            train,
            model,
            config,
            train["labels"],
            clean_concepts["train"],
            train["teacher"][:, selected],
            batch_size=settings["batch_size"],
            shuffle=True,
            generator=shuffle_rng,
        )
        best_nll = float("inf")
        for epoch in range(settings["epochs"]):
            model.train()
            total_loss = 0.0
            for inputs, labels, anchor, teacher in loader:
                inputs = model.prepare_inputs(inputs.float().to(device))
                labels = labels.to(device)
                anchor, teacher = anchor.to(device), teacher.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, concepts, energy = model(inputs)
                loss = F.cross_entropy(logits, labels)
                loss += settings["concept_anchor_weight"] * F.mse_loss(concepts, anchor)
                loss += settings["noise_energy_weight"] * energy
                if settings["stage"] == "finetune_projection":
                    loss += settings["teacher_alignment_weight"] * (
                        1 - cos_cubed(teacher, concepts).mean()
                    )
                loss.backward()
                # One shared norm covers the generator and trainable projection.
                torch.nn.utils.clip_grad_norm_(trainable, settings["grad_clip_norm"])
                optimizer.step()
                total_loss += loss.item() * len(labels)
            # Common random draws make validation comparisons less noisy and
            # do not consume the training RNG stream.
            with seeded(config["experiment"]["seed"] + 1000):
                validation = evaluate(
                    model, data["splits"]["val"], clean_concepts["val"], selected, config, device
                )
            record = {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(loader.dataset),
                "val": validation,
            }
            history.append(record)
            print(
                f"Epoch {epoch + 1}: loss={record['train_loss']:.4f}, val_acc={validation['accuracy']:.3f}, val_nll={validation['nll']:.4f}",
                flush=True,
            )
            if validation["nll"] < best_nll:
                best_nll = validation["nll"]
                best_cbm = deepcopy(model.cbm.state_dict())
                best_generator = deepcopy(model.generator.state_dict())
        model.cbm.load_state_dict(best_cbm)
        model.generator.load_state_dict(best_generator)

    model.eval().requires_grad_(False)
    refit_metrics = None
    if settings["stage"] in {"refit_head", "finetune_projection"}:
        cached = {}
        # Freeze one sampled concept vector per example for SAGA's fixed-feature table.
        # The head optimizer never sees fresh random features mid-iteration.
        with seeded(config["experiment"]["seed"] + 2000), torch.no_grad():
            for split_name in ("train", "val"):
                values = []
                split = data["splits"][split_name]
                loader = input_loader(split, model, config, batch_size=config["data"]["batch_size"])
                for (inputs,) in loader:
                    inputs = model.prepare_inputs(inputs.float().to(device))
                    _, concepts = predict(model, inputs)
                    values.append(concepts.cpu())
                cached[split_name] = torch.cat(values)
        refit_metrics = fit_saga_head(
            model.cbm.head,
            cached["train"],
            data["splits"]["train"]["labels"],
            cached["val"],
            data["splits"]["val"]["labels"],
            config["baseline"]["head"],
        )
    return history, refit_metrics
