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

    config["train_dataset"]["args"]["load_into_memory"] = config["use_memory"]
    train_dataset = initialize_config(config["train_dataset"])
    config["validation_dataset"]["args"]["spk2indx"] = train_dataset.get_spk2indx()
    val_dataset = initialize_config(config["validation_dataset"])
    test_dataset = initialize_config(config["test_dataset"])

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

    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=1,
        drop_last=True
    )

    model = initialize_config(config["model"])

    config["loss_function"]["args"]["n_speakers"] = len(train_dataset.spk2indx)
    config["loss_function"]["args"]["learnable_emb"] = config['optimizer']['update_emb']
    loss_function = initialize_config(config["loss_function"])


    trainable_params = []
    if config['optimizer']['update_emb']:
        trainable_params.append({'params':loss_function['spk'].parameters(), 'lr': config['optimizer']['lr_loss']})
    if config['optimizer']['update_spk']:
        trainable_params.append({'params':model.spk_stack.parameters(), 'lr': config['optimizer']['lr_spk']})
    if config['optimizer']['update_sep']:
        trainable_params.append({'params':model.sep_stack.parameters(), 'lr': config['optimizer']['lr_sep']})

    optimizer = torch.optim.Adam(
        params=trainable_params,
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
        validation_dataloader=valid_dataloader,
        test_dataloader=test_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-attention for Speech Enhancement")
    parser.add_argument(
        "-C", "--configuration",
        required=True,
        type=str,
        help="select config file for training <*.yaml>"
    )
    parser.add_argument(
        "-E", "--eval_before_train",
        action="store_true",
        help="evaluation on val and test set before training"
    )
    parser.add_argument(
        "-P", "--preloaded_model_path",
        type=str,
        help="preloaded_model_path"
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
    parser.add_argument(
        "-M", "--use_memory",
        action="store_true",
        help="use_memory_for_preloading_dataset"
    )
    parser.add_argument(
        "-D", "--debug_mode",
        action="store_true",
        help="The results will not stored in <exp/> in this mode"
    )
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert args.resume == False, "Resume conflict with preloaded model. Please use one of them."
    
    with open(args.configuration) as f:
        configuration = yaml.safe_load(f)
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    configuration["preloaded_model_path"] = args.preloaded_model_path
    configuration["eval_before_train"] = args.eval_before_train
    configuration["use_cpu"] = args.use_cpu
    configuration["only_inference"] = args.only_inference
    configuration["use_memory"] = args.use_memory
    configuration["debug_mode"] = args.debug_mode

    main(configuration)
    