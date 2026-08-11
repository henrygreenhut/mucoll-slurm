#!/usr/bin/env python3
"""Shared binary-PFN model, optimization, and validation engine.

Data-specific entry points provide two callbacks:

* ``train_batches(epoch)`` yields balanced padded ``(x, y, labels)`` batches.
* ``predict_validation(model)`` returns fixed validation labels and scores.

Everything that can change the fitted classifier lives here: model
construction, Adam and its learning-rate schedule, checkpoint/resume,
validation metrics, best-model selection, early stopping, and history.
"""

import csv
import itertools
import json
import os
import time

import numpy as np

import libtest_common as lc


CONFIG_SCHEMA_VERSION = 2
RUNTIME_CONFIG_KEYS = {"max_minutes", "progress_every"}


def per_unit_cross_entropy(labels, scores):
    """Per-unit two-class cross entropy from class-1 probabilities."""
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.where(labels == 1, scores, 1.0 - scores)
    return -np.log(probabilities)


def binary_cross_entropy(labels, scores):
    """Mean two-class cross entropy from class-1 probabilities."""
    return float(np.mean(per_unit_cross_entropy(labels, scores)))


def initial_state(select_metric):
    return {
        "epoch": 0,
        "max_val_auc": -1.0,
        "max_val_auc_epoch": -1,
        "min_val_loss": float("inf"),
        "min_val_loss_epoch": -1,
        "best_metric_value": (-1.0 if select_metric == "auc" else float("inf")),
        "best_epoch": -1,
        "done": False,
    }


def load_state(path, select_metric):
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    return initial_state(select_metric)


def save_state(path, state):
    with open(path, "w") as handle:
        json.dump(state, handle, indent=1)


def append_history(path, row):
    exists = os.path.isfile(path)
    fieldnames = list(row)
    if exists:
        with open(path, newline="") as handle:
            fieldnames = next(csv.reader(handle))
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_or_validate_config(path, config):
    """Create an immutable scientific config; permit runtime-only changes."""
    config = json.loads(json.dumps(config))
    if not os.path.isfile(path):
        with open(path, "w") as handle:
            json.dump(config, handle, indent=1, sort_keys=True)
        return
    with open(path) as handle:
        saved = json.load(handle)
    saved.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("gradient_accumulation_steps", 1)
    saved.setdefault("exclude_muons_above_gev", 0.0)
    config.setdefault("exclude_muons_above_gev", 0.0)
    saved.setdefault("exclude_muons", False)
    config.setdefault("exclude_muons", False)
    if saved.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(
            "{} is a legacy/incompatible run. Use a new --label.".format(path))
    mismatches = []
    for key in sorted(set(saved) | set(config)):
        if key in RUNTIME_CONFIG_KEYS:
            continue
        if saved.get(key) != config.get(key):
            mismatches.append(
                "  {}: saved={!r}, requested={!r}".format(
                    key, saved.get(key), config.get(key)))
    if mismatches:
        raise SystemExit(
            "Refusing to resume with a changed scientific configuration.\n"
            + "\n".join(mismatches) + "\nUse a new --label.")


def update_validation_state(state, val_auc, val_loss, val_loss_sem,
                            select_metric, min_delta, min_delta_sigma, epoch):
    """Track both extrema and decide whether the selection metric improved."""
    if val_auc > state["max_val_auc"]:
        state["max_val_auc"] = val_auc
        state["max_val_auc_epoch"] = epoch
    if val_loss < state["min_val_loss"]:
        state["min_val_loss"] = val_loss
        state["min_val_loss_epoch"] = epoch
    if select_metric == "loss":
        improved = (
            val_loss
            < state["best_metric_value"] - min_delta_sigma * val_loss_sem)
        candidate = val_loss
    else:
        improved = val_auc > state["best_metric_value"] + min_delta
        candidate = val_auc
    if improved:
        state["best_metric_value"] = candidate
        state["best_epoch"] = epoch
    return improved


def current_learning_rate(model):
    schedule = getattr(
        model.optimizer, "_learning_rate", model.optimizer.learning_rate)
    value = (
        schedule(model.optimizer.iterations) if callable(schedule) else schedule)
    return float(np.asarray(value.numpy() if hasattr(value, "numpy") else value))


def _build_model(config):
    train_kwargs = {
        "lr": config["lr"],
        "warmup_steps": config["warmup_steps"],
        "clipnorm": config["clipnorm"],
        "decay_steps": config["decay_steps"],
        "min_lr": config["min_lr"],
        "latent_dropout": config.get("latent_dropout", 0.0),
        "f_dropouts": config.get("f_dropout", 0.0),
        "phi_l2": config.get("phi_l2", 0.0),
        "f_l2": config.get("f_l2", 0.0),
    }
    if config["arch"] == "energyflow":
        if config["latent_scale"] == 1.0:
            return lc.build_pfn_energyflow(
                config["n_features"],
                phi_sizes=config["phi_sizes"],
                f_sizes=config["f_sizes"],
                jit_compile=config["jit"],
                **train_kwargs)
        return lc.build_pfn_energyflow_scaled(
            config["n_features"],
            config["latent_scale"],
            phi_sizes=config["phi_sizes"],
            f_sizes=config["f_sizes"],
            jit_compile=config["jit"],
            **train_kwargs)
    if config["arch"] == "local":
        return lc.build_pfn(
            config["n_features"],
            config["latent_scale"],
            phi_sizes=config["phi_sizes"],
            f_sizes=config["f_sizes"],
            jit_compile=config["jit"],
            **train_kwargs)
    raise ValueError("unknown PFN architecture {!r}".format(config["arch"]))


def accumulated_train_step(model, batches, accumulation_steps, tf):
    gradient_sums = [None] * len(model.trainable_variables)
    losses = []

    for x, y, _ in batches:
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = model.compiled_loss(
                tf.convert_to_tensor(y),
                predictions,
                regularization_losses=model.losses,
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            gradient = tf.convert_to_tensor(gradient)
            gradient_sums[index] = (
                gradient if gradient_sums[index] is None
                else gradient_sums[index] + gradient
            )
        losses.append(float(loss.numpy()))
        del x, y, predictions, loss, gradients

    if len(losses) != accumulation_steps:
        raise RuntimeError(
            "received {} microbatches; expected {}".format(
                len(losses), accumulation_steps))
    gradients_and_variables = [
        (gradient / float(accumulation_steps), variable)
        for gradient, variable in zip(
            gradient_sums, model.trainable_variables)
        if gradient is not None
    ]
    model.optimizer.apply_gradients(gradients_and_variables)
    return losses


def run_binary_pfn_training(config, train_batches, predict_validation,
                            start_time=None):
    """Fit one binary PFN using data-adapter callbacks.

    Returns ``(model, state, complete)``. ``complete`` is false only when the
    wall-clock budget was reached after a safely checkpointed epoch.
    """
    import tensorflow as tf

    required = {
        "result_dir", "n_features", "latent_scale", "phi_sizes", "f_sizes",
        "arch", "jit", "lr", "warmup_steps", "decay_steps", "min_lr",
        "clipnorm", "model_seed", "select_metric", "min_delta",
        "min_delta_sigma", "epochs", "patience", "min_epochs",
        "units_per_epoch", "batch_size", "max_minutes", "progress_every",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("training config is missing: {}".format(
            ", ".join(missing)))
    if config["batch_size"] < 2 or config["batch_size"] % 2:
        raise ValueError("balanced training requires an even batch size >= 2")
    if config["units_per_epoch"] % (config["batch_size"] // 2):
        raise ValueError(
            "units_per_epoch must be divisible by batch_size/2")

    if start_time is None:
        start_time = time.time()
    tf.keras.utils.set_random_seed(config["model_seed"])
    result_dir = config["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    state_path = os.path.join(result_dir, "state.json")
    history_path = os.path.join(result_dir, "history.csv")
    best_weights = os.path.join(result_dir, "best.weights.h5")
    best_auc_weights = os.path.join(result_dir, "best_auc.weights.h5")
    last_weights = os.path.join(result_dir, "last.weights.h5")
    accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    microbatches_per_epoch = (
        2 * config["units_per_epoch"] // config["batch_size"])
    if microbatches_per_epoch % accumulation_steps:
        raise ValueError(
            "microbatches per epoch must be divisible by "
            "gradient_accumulation_steps")
    steps_per_epoch = microbatches_per_epoch // accumulation_steps

    model = _build_model(config)
    print(
        "  XLA JIT: requested {} | model effective {} | optimizer effective {}"
        .format(
            config["jit"], getattr(model, "_jit_compile", None),
            bool(getattr(model.optimizer, "jit_compile", False))))
    if accumulation_steps > 1:
        print(
            "  gradient accumulation: {} microbatches x {} events "
            "= effective batch {}, {} optimizer updates/epoch".format(
                accumulation_steps, config["batch_size"],
                accumulation_steps * config["batch_size"],
                steps_per_epoch))
    if hasattr(model.optimizer, "build"):
        model.optimizer.build(model.trainable_variables)

    state = load_state(state_path, config["select_metric"])
    checkpoint_values = {
        "epoch": tf.Variable(0, dtype=tf.int64, trainable=False),
        "max_val_auc": tf.Variable(-1.0, dtype=tf.float64, trainable=False),
        "max_val_auc_epoch": tf.Variable(-1, dtype=tf.int64, trainable=False),
        "min_val_loss": tf.Variable(
            float("inf"), dtype=tf.float64, trainable=False),
        "min_val_loss_epoch": tf.Variable(
            -1, dtype=tf.int64, trainable=False),
        "best_metric_value": tf.Variable(
            -1.0 if config["select_metric"] == "auc" else float("inf"),
            dtype=tf.float64, trainable=False),
        "best_epoch": tf.Variable(-1, dtype=tf.int64, trainable=False),
    }
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer, **checkpoint_values)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(result_dir, "resume_checkpoint"),
        max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        try:
            status.assert_consumed()
        except (AssertionError, ValueError) as exc:
            raise SystemExit(
                "Checkpoint does not exactly match this model, optimizer, "
                "and state schema. Refusing a partial restore; use a new "
                "--label.\n{}".format(exc))
        for name, value in checkpoint_values.items():
            state[name] = (
                int(value.numpy()) if "epoch" in name else float(value.numpy()))
        print(
            "  resumed model + Adam from epoch {} "
            "(max val AUC {:.4f}, min val loss {:.4f})".format(
                state["epoch"], state["max_val_auc"], state["min_val_loss"]))
    elif state["epoch"] > 0:
        raise SystemExit(
            "state.json reports a resumed run but no full TensorFlow "
            "checkpoint exists; refusing to restore weights without Adam state")

    while not state["done"] and state["epoch"] < config["epochs"]:
        epoch = state["epoch"]
        lr_value = current_learning_rate(model)
        train_start = time.time()
        losses = []
        batches = iter(train_batches(epoch))
        for step in range(1, steps_per_epoch + 1):
            if accumulation_steps == 1:
                try:
                    x, y, _ = next(batches)
                except StopIteration:
                    raise RuntimeError(
                        "data adapter ended before optimizer update {}".format(
                            step))
                output = model.train_on_batch(x, y)
                losses.append(float(
                    output[0] if isinstance(output, (list, tuple)) else output))
            else:
                microbatches = itertools.islice(
                    batches, accumulation_steps)
                losses.extend(accumulated_train_step(
                    model, microbatches, accumulation_steps, tf))
            if (
                    config["progress_every"]
                    and (step == 1
                         or step % config["progress_every"] == 0
                         or step == steps_per_epoch)):
                print(
                    "  train batch {}/{}: mean loss {:.4f}, {:.0f}s".format(
                        step, steps_per_epoch, np.mean(losses),
                        time.time() - train_start),
                    flush=True)
        try:
            next(batches)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "data adapter yielded more than {} microbatches".format(
                    microbatches_per_epoch))
        if len(losses) != microbatches_per_epoch:
            raise RuntimeError(
                "recorded {} microbatch losses; expected {}".format(
                    len(losses), microbatches_per_epoch))
        train_seconds = time.time() - train_start

        val_start = time.time()
        labels, scores = predict_validation(model)
        val_seconds = time.time() - val_start
        val_auc = lc.auc_score(labels, scores)
        unit_losses = per_unit_cross_entropy(labels, scores)
        val_loss = float(np.mean(unit_losses))
        val_loss_sem = float(
            np.std(unit_losses, ddof=1) / np.sqrt(len(unit_losses)))

        state["epoch"] = epoch + 1
        new_max_auc = val_auc > state["max_val_auc"]
        improved = update_validation_state(
            state, val_auc, val_loss, val_loss_sem,
            config["select_metric"], config["min_delta"],
            config["min_delta_sigma"], epoch)
        if new_max_auc:
            model.save_weights(best_auc_weights)
        if improved:
            model.save_weights(best_weights)
        model.save_weights(last_weights)
        for name, value in checkpoint_values.items():
            value.assign(state[name])
        checkpoint_manager.save(checkpoint_number=state["epoch"])
        append_history(history_path, {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_loss_sem": val_loss_sem,
            "val_auc": val_auc,
            "learning_rate": lr_value,
            "train_seconds": round(train_seconds, 1),
            "val_seconds": round(val_seconds, 1),
            "seconds": round(train_seconds + val_seconds, 1),
        })
        save_state(state_path, state)
        print(
            "epoch {}: loss {:.4f} | val loss {:.4f} (SEM {:.4f}) | "
            "val AUC {:.4f}{} | lr {:.3g} | train {:.0f}s + val {:.0f}s"
            .format(
                epoch, np.mean(losses), val_loss, val_loss_sem, val_auc,
                " *" if improved else "", lr_value,
                train_seconds, val_seconds),
            flush=True)

        if lc.should_early_stop(
                state, config["patience"], config["min_epochs"]):
            state["done"] = True
            save_state(state_path, state)
            print(
                "early stop: no validation-{} improvement for {} epochs"
                .format(config["select_metric"], config["patience"]))
        if (
                config["max_minutes"] > 0
                and (time.time() - start_time) / 60.0
                > config["max_minutes"]):
            print("wall-clock limit reached -- checkpoint saved; "
                  "resubmit with the same --label to resume")
            return model, state, False

    if state["epoch"] >= config["epochs"]:
        state["done"] = True
        save_state(state_path, state)
    if os.path.isfile(best_weights):
        model.load_weights(best_weights)
    return model, state, True
