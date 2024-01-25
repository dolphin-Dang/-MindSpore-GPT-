import argparse

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from mindspore.train.model import Model
from mindspore import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.gpt import GPT, GPTWithLoss
from src.utils import GPTConfig, LearningRate

SEQ_LEN = 128
VOCAB_SIZE = 1928

def run_train():
    """train function"""
    # local config
    local_config = {
        "optimizer": "adam", # adam, lamb
        "epoch_size": 2,
        "data_path": "./dataset",
        "ckpt_path": "GPT_baseline.ckpt",
        "start_lr": 5e-4,
        "end_lr": 1e-6,
        "sink_size": 10
    }
    
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)

    config = GPTConfig(
        batch_size=64,
        seq_length=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        embedding_size=128,
        num_layers=4,
        num_heads=8,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=1.0,
        use_past=False,
    )
    gpt = GPT(config)
    gpt_with_loss = GPTWithLoss(gpt)

    ds = create_dataset(config.batch_size, data_path=local_config["data_path"])
    print("Dataset created.")
    
    epoch_num = local_config["epoch_size"]
    step_per_epoch = ds.get_dataset_size()

    lr = LearningRate(
        learning_rate=local_config["start_lr"],
        end_learning_rate=local_config["end_lr"],
        warmup_steps=int(step_per_epoch * epoch_num * 0.1),
        decay_steps=epoch_num * step_per_epoch,
    )

    decay_filter = (
        lambda x: "layernorm" not in x.name.lower() and "bias" not in x.name.lower()
    )
    params = gpt.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": other_params, "weight_decay": 0.0},
        {"order_params": params},
    ]

    if local_config["optimizer"] == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif local_config["optimizer"] == "adam":
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr)
    else:
        raise AsserError(f'No such optimizer {local_config["optimizer"]}!')

    callback_size = local_config["sink_size"]
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="GPT", config=config_ck)
    callback.append(ckpoint_cb)

    print(f"epoch num = {epoch_num}")
    print(f"callback size = {callback_size}")
    print(f"actual epoch num = {actual_epoch_num}")
    print(f"step per epoch = {step_per_epoch}")
    gpt_with_loss.set_train(True)

    # load pre-trained params
    if local_config["ckpt_path"] != None:
        param_dict = load_checkpoint(local_config["ckpt_path"])
        load_param_into_net(gpt_with_loss, param_dict)
        print(f'Pre-trained ckpt loaded : {local_config["ckpt_path"]}')
    
    model = Model(gpt_with_loss, optimizer=optimizer)
    model.train(
        actual_epoch_num,
        ds,
        callbacks=callback,
        dataset_sink_mode=True,
        sink_size=callback_size,
    )
    print("The End!")

if __name__ == "__main__":
    set_seed(2023)
    run_train()
