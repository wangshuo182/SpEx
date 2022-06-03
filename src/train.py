import argparse
import os

# import json5
import yaml
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from util.utils import initialize_config

# import warnings
# warnings.filterwarnings('error')

def main(config):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])

    train_dataset = initialize_config(config["train_dataset"])
    config["validation_dataset"]["args"]["spk2indx"] = train_dataset.get_spk2indx()
    val_dataset = initialize_config(config["validation_dataset"])

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"],
        drop_last=True
    )

    valid_dataloader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=1,
        drop_last=True
    )

    model = initialize_config(config["model"])

    config["loss_function"]["args"]["n_speakers"] = len(train_dataset.spk2indx)
    loss_function = initialize_config(config["loss_function"])

    optimizer = torch.optim.Adam(
        params=list(model.parameters()) + list(loss_function['spk'].parameters()),
        lr=config["optimizer"]["lr"]
    )

    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=config["resume"],
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-attention for Speech Enhancement")
    parser.add_argument(
        "-C", "--configuration",
        required=True,
        type=str,
        help="指定用于训练的配置文件 *.yaml"
    )
    parser.add_argument(
        "-O", "--omit_visualize_unprocessed_speech",
        action="store_true",
        help="每次实验开始时（首次或重启），会在验证集上计算基准性能（未处理时的）。 可以通过此选项跳过这个步骤。"
    )
    parser.add_argument(
        "-P", "--preloaded_model_path",
        type=str,
        help="预加载的模型路径。"
    )
    parser.add_argument(
        "-R", "--resume",
        action="store_true",
        help="Resume experiment from latest checkpoint."
    )
    parser.add_argument(
        "-U", "--use_cpu",
        action="store_true",
        help="cpu only"
    )
    parser.add_argument(
        "-I", "--only_inference",
        action="store_true",
        help="only_inference_in_cv_dataset"
    )
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert args.resume == False, "Resume conflict with preloaded model. Please use one of them."

    # configuration = json5.load(open(args.configuration))
    
    with open(args.configuration) as f:
        configuration = yaml.safe_load(f)
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    configuration["preloaded_model_path"] = args.preloaded_model_path
    configuration["omit_visualize_unprocessed_speech"] = args.omit_visualize_unprocessed_speech
    configuration["use_cpu"] = args.use_cpu
    configuration["only_inference"] = args.only_inference

    main(configuration)
    