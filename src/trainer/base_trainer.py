import time
from pathlib import Path
import os

import yaml
import numpy as np
import torch
from util import visualization
from util.utils import prepare_empty_dir, ExecutionTime
from datetime import datetime

class BaseTrainer:
    def __init__(self, config, resume: bool, model, loss_function, optimizer):
        self.n_gpu = 0 if config["use_cpu"] else torch.cuda.device_count()
        self.device = self._prepare_device(self.n_gpu, cudnn_deterministic=config["cudnn_deterministic"])
        self.only_inference = config["only_inference"]
        self.eval_before_train = config["eval_before_train"]
        self.debug_mode = config["debug_mode"]
        if self.debug_mode: config["output_dir"] = "cache"
        # print("n_gpu:{}".format(self.n_gpu))
        
        self.load_spk_emd = config['trainer']['load_spk_emd']
        self.load_spk_net = config['trainer']['load_spk_net']
        self.load_sep_net = config['trainer']['load_sep_net']
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model.to(self.device)
        # TODO
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        # Trainer
        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.validation_config = config["trainer"]["validation"]
        self.train_config = config["trainer"].get("train", {})
        self.validation_interval = self.validation_config["interval"]
        self.find_max = self.validation_config["find_max"]
        self.validation_custom_config = self.validation_config["custom"]
        self.train_custom_config = self.train_config.get("custom", {})

        # The following args is not in the config file. We'll update them in later if resume is True.
        self.start_epoch = 1
        self.best_score = -np.inf if self.find_max else np.inf
        self.output_dir = Path(config["output_dir"]).expanduser().absolute() / config["experiment_name"]
        if resume:
            timestamp = config["resume_timestamp"]
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoints_dir = self.output_dir / timestamp / "checkpoints"
        self.logs_dir = self.output_dir / timestamp / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

        self.writer = visualization.writer(self.logs_dir.as_posix())
        self.writer.add_text(
            tag="Configuration",
            text_string=f"<pre>  \n{yaml.dump(config, sort_keys=False)}  \n</pre>",
            global_step=1
        )

        if resume: self._resume_checkpoint()
        if config["preloaded_model_path"]: self._preload_model(Path(config["preloaded_model_path"]))

        print("Configurations are as follows: ")
        print(yaml.dump(config, sort_keys=False))

        with open((self.output_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.yaml").as_posix(), "w") as handle:
            yaml.dump(config, handle, sort_keys=False)

        self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload *.pth file of the model at the start of the current experiment.

        Args:
            model_path(Path): the path of the *.pth file
        """
        model_path = model_path.expanduser().absolute()
        assert model_path.exists(), f"Preloaded *.pth file is not exist. Please check the file path: {model_path.as_posix()}"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=self.device)
        if model_path.suffix == '.tar':
            model_checkpoint = model_checkpoint['model']
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_checkpoint)
        else:
            self.model.load_state_dict(model_checkpoint)

        print(f"Model preloaded successfully from {model_path.as_posix()}.")

    def _resume_checkpoint(self):
        """Resume experiment from latest checkpoint.
        Notes:
            To be careful at Loading model. if model is an instance of DataParallel, we need to set model.module.*
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <output_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <output_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):  # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()

        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of the model. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        # Use model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need re-migrate model back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.model.to(self.device)

    @staticmethod
    def _prepare_device(n_gpu: int, cudnn_deterministic=False):
        """Choose to use CPU or GPU depend on "n_gpu".
        Args:
            n_gpu(int): the number of GPUs used in the experiment.
                if n_gpu is 0, use CPU;
                if n_gpu > 1, use GPU.
            cudnn_deterministic (bool): repeatability
                cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
        """
        if n_gpu == 0:
            print("Using CPU in the experiment.")
            device = torch.device("cpu")
        else:
            if cudnn_deterministic:
                print("Using CuDNN deterministic mode in the experiment.")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            device = torch.device("cuda:0")

        return device

    def _is_best(self, score, find_max=True):
        """Check if the current model is the best model
        """
        if find_max and score >= self.best_score:
            self.best_score = score
            return True
        elif not find_max and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _transform_pesq_range(pesq_score):
        """transform [-0.5 ~ 4.5] to [0 ~ 1]
        """
        return (pesq_score + 0.5) / 5

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def _on_train_start(self):
        pass

    def _on_validation_epoch_start(self):
        pass

    def train(self):
        self._on_train_start()
        if self.only_inference:
            print("Model is preloaded, Inference is in progress in cv dataset...")
            timer = ExecutionTime()
            self._set_models_to_eval_mode()
            # self._on_validation_epoch_start()
            score = self._inference()
            print(f"[{timer.duration()} seconds] Done the inference.")
            return
        
        if self.eval_before_train:
            print("======validation and test before epoch 0 or checkpoint======")
            timer = ExecutionTime()
            self._set_models_to_eval_mode()
            self._on_validation_epoch_start()
            val_score = self._validation_epoch(0)
            print(f"[{timer.duration()} seconds] Done the Inference on Val set.")
            test_score = self._inference()
            print(f"[{timer.duration()} seconds] Done the inference on Test set.")
            print(f"[val: {val_score:.2f}] [test: {test_score:.2f}] scores yield before traning.")

        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)
            # self.scheduler.step()

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch)

            if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                print(f"[{timer.duration()} seconds] Training is over, Validation is in progress...")

                self._set_models_to_eval_mode()
                self._on_validation_epoch_start()
                val_score = self._validation_epoch(epoch)

                if self._is_best(val_score, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)
            
            test_score = self._test_epoch(epoch)
            
            print(f"[{timer.duration()} seconds] End this epoch.")
            print(f"[val: {val_score:.2f}] [test: {test_score:.2f}] scores yield in this epoch.")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError

    def _test_epoch(self, epoch):
        raise NotImplementedError


