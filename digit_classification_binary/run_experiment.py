import sys
import os
import torch
import torchvision
import json
import gzip
import yaml
import argparse
import pathlib
import random
import numpy as np

from operator import attrgetter
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import mnist_data
import ardis_data
import mnist_models

from utilities import get_datetime
from utilities import JsonNumericEncoder


def get_attr(module_attr_str):
    """
    """
    module_attr_str = module_attr_str.split(".")
    module, attr = module_attr_str[0], module_attr_str[1:]
    module = sys.modules[module]
    attr = ".".join(attr)
    attr = attrgetter(attr)(module)

    return attr


def load_configs(args):
    """
    """
    dataset_ids = args.dataset_ids
    model_ids = args.model_ids
    data_configs = dict()
    model_configs = dict()
    if args.data_configs is not None:
        for dataset_id, data_config_path in zip(dataset_ids, args.data_configs[:len(dataset_ids)]):
            with open(data_config_path, 'r') as f:
                data_configs[dataset_id] = yaml.load(f, Loader=yaml.FullLoader)
    if args.model_configs is not None:
        for model_id, model_config_path in zip(model_ids, args.model_configs[:len(model_ids)]):
            with open(model_config_path, 'r') as f:
                model_configs[model_id] = yaml.load(f, Loader=yaml.FullLoader)

    return data_configs, model_configs


def load_stats(path):
    """
    """
    # Load stats
    with open(path, 'r') as f:
        stats = json.load(f) #, object_pairs_hook=JsonNumericDecoder)

    return stats


def create_model(model_config, model_id, ckpt_path=None):
    """
    """
    # model_class = get_attr(model_config["model_class"])
    model_class = get_attr(model_config["model_class"]["name"])
    model_kwargs = model_config["model_class"]["kwargs"] if "kwargs" in model_config["model_class"] else dict()
    # model = model_class(model_id)
    model = model_class(model_id, **model_kwargs)
    print("{} Model {} created.".format(get_datetime(), model_class))

    # if "ckpt_path" in model_config:
    #     ckpt_path = model_config["ckpt_path"][model_id]
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("{} Checkpoint loaded from {}.".format(get_datetime(), ckpt_path))

    return model


def load_model(model_id):
    """
    """
    if model_id in model_id_group1:
        model = torch.hub.load("ultralytics/yolov5", model_id, force_reload=True)
    if model_id in model_id_group2:
        model = eval("torchvision.models.detection.{}(pretrained=True)".format(model_id))
        model.eval()

    return model


def save_model(model, optimizer, save_path):
    """
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, save_path)


def save_stats(stats, save_path):
    """
    """
    with open(save_path, 'w') as f:
        json.dump(stats, f, cls=JsonNumericEncoder)


def create_dataloader(data_config):
    """
    """
    # dataset_class = get_attr(data_config["dataset_class"])
    dataset_class = get_attr(data_config["dataset_class"]["name"])
    dataset_kwargs = data_config["dataset_class"]["kwargs"] if "kwargs" in data_config["dataset_class"] else dict()
    batch_size = data_config["batch_size"]
    shuffle = data_config["shuffle"]

    transform_list = []
    for t in data_config["transform"]:
        transform = get_attr(t["name"])(**t["kwargs"]) if "kwargs" in t else get_attr(t["name"])()
        transform_list.append(transform)
    transform = transforms.Compose(transform_list)
    # dataset = dataset_class(data_config, transform=transform)
    dataset = dataset_class(data_config, transform=transform, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print("{} Dataset {} and dataloader created.".format(get_datetime(), dataset_class))

    return dataloader


def configure_loss_function(model_config, model):
    """
    """
    loss_function_class = get_attr(model_config["loss_function"]["name"])
    if "RidgeClassifierLoss" in model_config["loss_function"]["name"]:
        loss_function = loss_function_class(model)
    else:
        loss_function = loss_function_class()

    return loss_function


def create_optimizer_and_lr_scheduler(model, model_config):
    """
    """
    optimizer_class = get_attr(model_config["optimizer"]["name"])
    optimizer = optimizer_class(model.parameters(), **model_config["optimizer"]["kwargs"])

    lr_scheduler = None
    if "lr_scheduler" in model_config:
        lr_scheduler_class = get_attr(model_config["lr_scheduler"]["name"])
        lr_scheduler = lr_scheduler_class(optimizer, **model_config["lr_scheduler"]["kwargs"])

    return optimizer, lr_scheduler


def _train_on_batch(model, batch, optimizer, device, loss_function):
    """
    """
    optimizer.zero_grad(set_to_none=True)
    image, label = batch["image"].float().to(device), batch["label"].to(device)
    output = model.forward(image)
    try:
        loss = loss_function(output, label)
    except:
        # torch.nn.CrossEntropyLoss expects int while torch.nn.BCELoss expects float
        if label.dtype in [torch.int32, torch.int64]:
            label = label.float()
        elif label.dtype in [torch.float32, torch.float64]:
            label = label.int()
        loss = loss_function(output, label)
    if hasattr(model, "regularization"):
        regularization = model.regularization()
        loss += regularization
        batch["regularization"] = regularization.detach().cpu().item()
    loss.backward()
    optimizer.step()

    batch["output"] = output.detach().cpu().numpy()
    batch["loss"] = loss.detach().cpu().item()

    return batch


def _evaluate_on_batch(model, batch, device, loss_function):
    """
    """
    model.eval()
    with torch.no_grad():
        image, label = batch["image"].float().to(device), batch["label"].to(device)
        output = model.forward(image)
        try:
            loss = loss_function(output, label)
        except:
            if label.dtype in [torch.int32, torch.int64]:
                label = label.float()
            elif label.dtype in [torch.float32, torch.float64]:
                label = label.int()
            loss = loss_function(output, label)
        # _, prediction = torch.max(output, 1)
        prediction = model.output_to_target(output)
        accuracy = (prediction == label).sum().detach().item() / label.size(0)

    batch["output"] = output.detach().cpu().numpy()
    batch["prediction"] = prediction.detach().cpu().numpy()
    batch["loss"] = loss.detach().cpu().item()
    batch["accuracy"] = accuracy

    return batch


class CKPTSavingHook(object):
    def __init__(self, ckpt_saving_hook_template):
        """
        Initialize self.status which is a dict of form {"ckpt_id_type": str, "life": (epoch, batch), "ckpt_id": [ckpt_id: val_acc]}
        """
        self.status = ckpt_saving_hook_template
        if self.status["ckpt_id_type"] == "epoch":
            self.status["ckpt_id"] = {(ckpt_id["epoch"], ckpt_id["batch"]): None for ckpt_id in ckpt_saving_hook_template["ckpt_id"]}
        elif self.status["ckpt_id_type"] == "val_acc":
            self.status["ckpt_id"] = {ckpt_id["val_acc"]: None for ckpt_id in ckpt_saving_hook_template["ckpt_id"]}
        else:
            raise ValueError("ckpt_id_type should be either 'epoch' or 'val_acc'.")
        self.status["valid"] = True
        print("{} CKPTSavingHook created: {}.".format(get_datetime(), self.status))

    def __call__(
            self,
            model,
            val_dataloader,
            loss_function,
            optimizer,
            lr_scheduler,
            device,
            save_path,
            num_val_batches,
            ckpt_saving_hook,
            epoch,
            num_epochs,
            batch,
            num_train_batches,
            train_loss,
            train_reg,
            val_loss,
            min_loss,
            val_acc
        ):
        if epoch >= self.status["life"]["epoch"] and batch >= self.status["life"]["batch"]:
            self.status["valid"] = False
            return
        if "batch" in self.status["call_freq"] and batch % self.status["call_freq"]["batch"] != 0:
            return
        if epoch % self.status["call_freq"]["epoch"] != 0:
            return
    
        if self.status["ckpt_id_type"] == "epoch":
            if (epoch, batch) not in self.status["ckpt_id"]:
                return
    
        val_loss = 0.0
        val_acc = 0.0
        for _, val_batch in enumerate(val_dataloader):
            val_batch = _evaluate_on_batch(model, val_batch, device, loss_function=loss_function)
            val_loss += val_batch["loss"] / num_val_batches
            val_acc += val_batch["accuracy"] / num_val_batches
        if val_loss < min_loss:
            min_loss = val_loss
        print("{} Epoch {}/{}, batch {}/{}, lr {:.8f}, train loss {:.8f}, reg {:.8f}, val loss/min {:.8f}/{:.8f}, acc {:.8f}".format(
            get_datetime(), epoch, num_epochs, batch, num_train_batches, lr_scheduler._last_lr[0], train_loss, train_reg, val_loss, min_loss, val_acc))
    
        ckpt_id = None
        if self.status["ckpt_id_type"] == "epoch":
            ckpt_id = (epoch, batch)
            save_path = "{}.ep{}_b{}".format(save_path, epoch, batch)
        else:
            sorted_keys = np.sort(list(self.status["ckpt_id"].keys()))
            indices = np.where(sorted_keys <= val_acc)[0]
            if indices.size > 0:
                ckpt_id = sorted_keys[indices[-1]]
                save_path = "{}.val_acc{}".format(save_path, ckpt_id)
    
        if ckpt_id in self.status["ckpt_id"] and self.status["ckpt_id"][ckpt_id] is None:
            print("{} ckpt saved to {}.".format(get_datetime(), save_path))
            self.status["ckpt_id"][ckpt_id] = val_acc
            save_model(model, optimizer, save_path)
    
        return

def _train(
        model_config,
        model,
        train_dataloader,
        val_dataloader,
        loss_function,
        optimizer,
        lr_scheduler,
        device,
        save_path,
        print_batch_freq=None,
        print_epoch_freq=1
    ):
    """
    """
    num_epochs = model_config["num_epochs"]
    save_ckpt = model_config["save_ckpt"]
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)
    model.to(device)
    model.train()

    min_loss = 1e5
    ckpt_saving_hook = None
    if "ckpt_saving_hook" in model_config:
        ckpt_saving_hook = CKPTSavingHook(model_config["ckpt_saving_hook"]) # !!!
    stats = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_reg = 0.0

        for i, train_batch in enumerate(train_dataloader):
            train_batch = _train_on_batch(model, train_batch, optimizer, device, loss_function=loss_function)
            train_loss += train_batch["loss"] / num_train_batches
            if "regularization" in train_batch:
                train_reg += train_batch["regularization"] / num_train_batches
            if print_batch_freq and i % print_batch_freq == 0:
                print("{} Epoch {}/{}, batch {}/{}, lr {}, batch loss/cum {:.8f}/{:.8f}".format(
                        get_datetime(), epoch, num_epochs, i, num_train_batches, lr_scheduler.get_lr(), train_batch["loss"], train_loss))
            # !!!
            if ckpt_saving_hook and ckpt_saving_hook.status["valid"]:
                ckpt_saving_hook(
                    model,
                    val_dataloader,
                    loss_function,
                    optimizer,
                    lr_scheduler,
                    device,
                    save_path,
                    num_val_batches,
                    ckpt_saving_hook,
                    epoch,
                    num_epochs,
                    i,
                    num_train_batches,
                    train_loss,
                    train_reg,
                    val_loss,
                    min_loss,
                    val_acc
                )

        for _, val_batch in enumerate(val_dataloader):
            val_batch = _evaluate_on_batch(model, val_batch, device, loss_function=loss_function)
            val_loss += val_batch["loss"] / num_val_batches
            val_acc += val_batch["accuracy"] / num_val_batches

        # lr_scheduler.step(val_loss)
        if lr_scheduler is not None:
            lr_scheduler.step()

        if val_loss < min_loss:
            min_loss = val_loss
            if save_ckpt:
                save_model(model, optimizer, save_path)

        if print_epoch_freq and epoch % print_epoch_freq == 0:
            print("{} Epoch {}/{}, lr {:.8f}, train loss {:.8f}, reg {:.8f}, val loss/min {:.8f}/{:.8f}, acc {:.8f}".format(
                    get_datetime(), epoch, num_epochs, lr_scheduler._last_lr[0], train_loss, train_reg, val_loss, min_loss, val_acc))

    if save_ckpt:
        print("{} Model saved to {}.".format(get_datetime(), save_path))

    stats["train"]["loss"] = train_loss
    stats["val"]["loss"] = val_loss

    return stats


def _evaluate(
        data_config,
        model_config,
        model,
        dataloader,
        loss_function,
        device,
        save_path,
    ):
    """
    """
    batch_size = data_config["batch_size"]
    dim_output = model.dim_output
    num_batches = len(dataloader)
    save_prediction = model_config["save_prediction"]
    model.to(device)
    model.eval()

    loss = 0.0
    acc = 0.0
    stats = {
        "output": np.zeros((num_batches * batch_size, dim_output)),
        "prediction": np.zeros(num_batches * batch_size),
        "loss": 0.0, 
        "accuracy": 0.0,
    }

    for i, batch in enumerate(dataloader):
        batch = _evaluate_on_batch(model, batch, device, loss_function=loss_function)
        loss += batch["loss"] / num_batches
        acc += batch["accuracy"] / num_batches
        if len(batch["output"].shape) < 2:  # LogisticRegressionMNIST forward() returns (N, ) instead of (N, 1) to be compatible with mnist_models.BinaryClassificationL2Loss
            batch["output"] = batch["output"][..., None]
        stats["output"][i*batch_size : (i+1)*batch_size] = batch["output"]
        stats["prediction"][i*batch_size : (i+1)*batch_size] = batch["prediction"]

    print("{} Loss: {:.8f}, acc: {:.8f}".format(get_datetime(), loss, acc))
    stats["loss"] = loss
    stats["accuracy"] = acc

    if save_prediction:
        save_stats(stats, save_path)
        print("{} Prediction saved to {}.".format(get_datetime(), save_path))
    stats.pop("output")
    stats.pop("prediction")

    return stats


def _collect_dataset_stats(data_config, dataloader):
    """
    """
    batch_size = data_config["batch_size"]
    num_batches = len(dataloader)
    stats = {
        "image_means": [],
        "image_stds": [],
    }

    for i, batch in enumerate(dataloader):
        stats["image_means"].append(torch.mean(batch["image"], (0, 2, 3))) # mean along all but channel axis
        stats["image_stds"].append(torch.std(batch["image"], (0, 2, 3))) # std along all but channel axis

    stats["image_means"] = torch.mean(torch.stack(stats["image_means"], dim=0), 0)
    stats["image_stds"] = torch.std(torch.stack(stats["image_stds"], dim=0), 0)

    return stats


def get_training_config_str(dataset_id, model_id, data_config, model_config, seed):
    """
    """
    loss_function_lut = {"torch.nn.CrossEntropyLoss": "ce", "mnist_models.BinaryClassificationL2Loss": "l2", "torch.nn.BCELoss": "bce"}
    optimizer_lut = {"torch.optim.SGD": "sgd", "torch.optim.RMSprop": "rmsprop", "torch.optim.Adam": "adam"}
    lr_scheduler_lut = {"torch.optim.lr_scheduler.StepLR": "steplr", "torch.optim.lr_scheduler.ReduceLROnPlateau": "reduclronp"}
    config_str = "{}_{}_{}_{}{}_ep{}_bs{}".format(
        dataset_id,
        model_id,
        loss_function_lut[model_config["loss_function"]["name"]],
        optimizer_lut[model_config["optimizer"]["name"]],
        model_config["optimizer"]["kwargs"]["lr"],
        model_config["num_epochs"],
        data_config["train"]["batch_size"],
    )
    if seed is not None:
        config_str += "_s{}".format(seed)

    return config_str


def get_evaluation_config_str(dataset_id, model_id, model_config, seed):
    """
    """
    loss_function_lut = {"torch.nn.CrossEntropyLoss": "ce"}
    config_str = "{}_{}".format(
        dataset_id,
        model_id,
        # loss_function_lut[model_config["loss_function"]["name"]],
    )
    if seed is not None:
        config_str += "_s{}".format(seed)

    return config_str


def print_stats(stats, dataset_ids, model_ids, ignore_intermediate_ckpt=True, ljust=30):
    """
    """
    stats_to_print = OrderedDict({"loss": "ce", "accuracy": "acc"})
    print("\n{}".format(get_datetime()))
    first_row = "\n" + "model".ljust(ljust)
    for stats_name in stats_to_print:
        for dataset_id in dataset_ids:
            first_row += "{} {}".format(dataset_id, stats_to_print[stats_name]).ljust(ljust)
    print(first_row)
    for model_id in model_ids:
        row = model_id.ljust(ljust)
        for stats_name in stats_to_print:
            for dataset_id in dataset_ids:
                _stats = []
                for run in stats[dataset_id][model_id]["runs"]:
                    if ignore_intermediate_ckpt and "ckpt_id" in run:
                        continue
                    _stats.append(run[stats_name])
                row += "{:.6f}/{:.6f}".format(np.mean(_stats), np.std(_stats)).ljust(ljust)
        print(row)


def collect_dataset_stats(data_config, dataset_key="test", gpu=0):
    """
    """
    # Create dataloader
    dataloader = create_dataloader(data_config[dataset_key])

    device = "cuda:{}".format(gpu)

    # Collect stats
    stats = _collect_dataset_stats(data_config[dataset_key], dataloader)

    return stats


def train(data_config, model_config, dataset_id, model_id, experiment_id, gpu=0, seed=None):
    """
    """
    # Create model
    model = create_model(model_config, model_id)
    # Move the model to GPU before constructing optimizers, as noted in 
    # https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783?u=shaibagon
    model.to("cuda:{}".format(gpu))

    # Create dataloader
    train_dataloader = create_dataloader(data_config["train"])
    val_dataloader = create_dataloader(data_config["val"])

    # Configure loss function
    loss_function = configure_loss_function(model_config, model)

    # Create optimizer and learning rate scheduler
    optimizer, lr_scheduler = create_optimizer_and_lr_scheduler(model, model_config)

    device = "cuda:{}".format(gpu)

    training_config_str = get_training_config_str(dataset_id, model_id, data_config, model_config, seed)
    if model_config["save_dir"]:
        save_path = os.path.join(model_config["save_dir"], "model_{}.pt".format(training_config_str))
    else:
        save_path = os.path.join("./{}".format(experiment_id), "ckpt", "model_{}.pt".format(training_config_str))
    print("{} Training config str: {}".format(get_datetime(), training_config_str))


    # Train
    stats = _train(
        model_config,
        model,
        train_dataloader,
        val_dataloader,
        loss_function,
        optimizer,
        lr_scheduler,
        device,
        save_path,
        print_batch_freq=None, # !!!
        # print_batch_freq=10,
        print_epoch_freq=1
    )

    stats["dataset_id"] = dataset_id
    stats["model_id"] = model_id

    return stats


def evaluate(data_config, model_config, dataset_id, model_id, experiment_id, gpu=0, seed=None):
    """
    """
    stats = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "runs": [],
    }
    for ckpt in model_config["ckpt"][model_id]:
        # Create model
        model = create_model(model_config, model_id, ckpt_path=ckpt["ckpt_path"])

        # Create dataloader
        dataloader = create_dataloader(data_config["test"])

        # Configure loss function
        loss_function = configure_loss_function(model_config, model)

        device = "cuda:{}".format(gpu)

        evaluation_config_str = get_evaluation_config_str(dataset_id, model_id, model_config, ckpt["seed"])
        save_path = os.path.join("./{}".format(experiment_id), "prediction_{}.json".format(evaluation_config_str))
        print("{} Evaluation config str: {}".format(get_datetime(), evaluation_config_str))

        # Evaluate
        _stats = _evaluate(
            data_config["test"],
            model_config,
            model,
            dataloader,
            loss_function,
            device,
            save_path,
        )
        _stats["seed"] = ckpt["seed"]
        if "ckpt_id" in ckpt:
            _stats["ckpt_id"] = ckpt["ckpt_id"]
        stats["runs"].append(_stats)

    return stats


def run_training(args):
    """
    """
    seed = args.seed
    dataset_ids = args.dataset_ids
    model_ids = args.model_ids
    experiment_id = args.experiment_id
    gpu = args.gpu
    data_configs, model_configs = load_configs(args)
    stats = dict()

    for dataset_id in dataset_ids:
        print("\n\n\n{} Training models on {}...".format(get_datetime(), dataset_id))
        stats[dataset_id] = dict()
        for model_id in model_ids:
            print("\n{} Training model {}...".format(get_datetime(), model_id))
            data_config = data_configs[dataset_id]
            model_config = model_configs[model_id]["train"]
            stats[dataset_id][model_id] = train(data_config, model_config, dataset_id, model_id, experiment_id, gpu=gpu, seed=seed)
    

def run_evaluation(args):
    """
    """
    seed = args.seed
    dataset_ids = args.dataset_ids
    model_ids = args.model_ids
    experiment_id = args.experiment_id
    gpu = args.gpu
    if args.stats_path:
        stats = load_stats(args.stats_path)
    else:
        stats = dict()
    data_configs, model_configs = load_configs(args)
    save_path = os.path.join("./{}".format(experiment_id), "stats_{}_s{}.json".format(experiment_id, seed))

    for dataset_id in dataset_ids:
        print("\n\n\n{} Evaluating models on {}...".format(get_datetime(), dataset_id))
        if dataset_id not in stats:
            stats[dataset_id] = dict()
        for model_id in model_ids:
            if dataset_id in stats and model_id in stats[dataset_id]:
                print("\n{} Stats of model {} exists, skipped...".format(get_datetime(), model_id))
            else:
                print("\n{} Evaluating model {}...".format(get_datetime(), model_id))
                data_config = data_configs[dataset_id]
                model_config = model_configs[model_id]["evaluate"]
                stats[dataset_id][model_id] = evaluate(data_config, model_config, dataset_id, model_id, experiment_id, gpu=gpu, seed=seed)

    print_stats(stats, dataset_ids, model_ids)
    save_stats(stats, save_path)
    print("{} Statistics saved to {}.".format(get_datetime(), save_path))
    

def run_dataset_analysis(args):
    """
    """
    dataset_ids = args.dataset_ids
    experiment_id = args.experiment_id
    gpu = args.gpu
    data_configs, model_configs = load_configs(args)
    stats = dict()

    for dataset_id in dataset_ids:
        print("\n\n\n{} Running analysis on {}...".format(get_datetime(), dataset_id))
        data_config = data_configs[dataset_id]
        stats[dataset_id] = collect_dataset_stats(data_config, dataset_key="test", gpu=gpu)
        print(stats[dataset_id])
    

def run_experiments(args):
    """
    """
    seed = args.seed
    mode = args.mode

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("{} Seed set to {}.".format(get_datetime(), seed))

    if mode == "train":
        run_training(args)
    elif mode == "evaluate":
        run_evaluation(args)
    elif mode == "collect_dataset_stats":
        run_dataset_analysis(args)
    else:
        raise ValueError("Mode must be either 'train' or 'evaluate', but got '{}'.".format(mode))


def run():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--dataset-ids", nargs="+", default=[None], type=str, help="Dataset identification strings.")
    parser.add_argument("--model-ids", nargs="+", default=[None], type=str, help="Model identification strings.")
    parser.add_argument("--experiment-id", type=str, help="Experiment identification string.")
    parser.add_argument("--data-configs", nargs="+", default=[None], type=pathlib.Path, help="Paths to YAML config file for data.")
    parser.add_argument("--model-configs", nargs="+", default=[None], type=pathlib.Path, help="Paths to YAML config file for model.")
    parser.add_argument("--mode", type=str, help="Either 'train' or 'evaluate'.")
    parser.add_argument("--gpu", type=str, help="GPU to be used.")
    parser.add_argument("--stats-path", default=None, type=pathlib.Path, help="Paths to JSON statistics file.")
    args = parser.parse_args()

    run_experiments(args)


def main():
    run()


if __name__ == "__main__":
    main()
