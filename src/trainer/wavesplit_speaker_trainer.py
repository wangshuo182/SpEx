import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from kmeans_pytorch import kmeans, kmeans_predict

from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from util.utils import compute_SDR

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader, test_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def _on_train_start(self):
        self.loss_function["spk"] = self.loss_function["spk"].to(self.device)

    def _on_validation_epoch_start(self):
        self.loss_function["spk"] = self.loss_function["spk"].to(self.device)

    def _train_epoch(self, epoch):
        loss_total = 0.0
        loss_step_sum = {'spk':0.0, 'sep':0.0, 'all':0.0}

        for iter, (mixture, oracle_s, oracle_ids, _) in enumerate(tqdm(self.train_dataloader, desc="Training")):
            mixture = mixture.to(self.device)
            oracle_s = oracle_s.to(self.device)
            oracle_ids = oracle_ids.to(self.device)
            b, n_spk, frames = oracle_s.size()

            self.optimizer.zero_grad()

            net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

            spk_vectors = net.get_speaker_vectors(mixture)
            b, n_spk, embed_dim, frames = spk_vectors.size()
            spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixture)
            spk_loss, reordered = self.loss_function["spk"](spk_vectors, spk_activity_mask, oracle_ids)

            reordered = reordered.mean(-1) # take centroid
            # reordered = self.loss_function["spk"].spk_embeddings[oracle_ids] #.to(self.device)
            # spk_loss = torch.Tensor([0]).to(self.device)

            # separated = net.split_waves(mixture, reordered)

            # if net.sep_stack.return_all:
            #     n_layers = len(separated)
            #     separated = torch.stack(separated).transpose(0, 1)
            #     separated = separated.reshape(
            #         b * n_layers, n_spk, frames
            #     )  # in validation take only last layer
            #     oracle_s = (
            #         oracle_s.unsqueeze(1).repeat(1, n_layers, 1, 1).reshape(b * n_layers, n_spk, frames)
            #     )
            
            sep_loss = 0#self.loss_function["sep"](separated, oracle_s).mean()
            loss = sep_loss + spk_loss
            
            loss.backward()
            all_parameters = list(net.parameters()) + list(self.loss_function["spk"].parameters())
            param_norm = torch.nn.utils.clip_grad_norm_(all_parameters, 5)
            self.optimizer.step()

            loss_total += loss.item()
            loss_step_sum['spk'] += spk_loss.item()
            # loss_step_sum['sep'] += sep_loss.item()
            # loss_step_sum['all'] += loss.item()

            if iter % 10 == 0 and iter > 0 :
                self.writer.add_scalars(f"Train/Loss", {
                    'epoch_spk_{}'.format(epoch): loss_step_sum['spk'] / 10,
                    # 'epoch_sep_{}'.format(epoch): loss_step_sum['sep'] / 10,
                    # 'epoch_all_{}'.format(epoch): loss_step_sum['all'] / 10
                }, iter)
                for i in loss_step_sum.keys(): loss_step_sum[i] = 0.0 

            # if i == 0:
                # self.writer.add_figure(f"Train_Tensor/Mixture", self.image_grad(mixture_mag.cpu()), epoch)
                # self.writer.add_figure(f"Train_Tensor/Target", self.image_grad(target_mag.cpu()), epoch)
                # self.writer.add_figure(f"Train_Tensor/Enhanced", self.image_grad(enhanced_mag.detach().cpu()), epoch)
                # self.writer.add_figure(f"Train_Tensor/Ref", self.image_grad(reference.cpu()), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)

        sdr_c_m = []  # Clean and mixture
        sdr_c_e = []  # Clean and enhanced

        self.intra_avg_distance = []
        self.inter_spk_distance = []
        self.sdr_improve = []

        for i, (mixture, oracle_s, oracle_ids, filename) in tqdm(enumerate(self.validation_dataloader)):
            b, n_spk, frames = oracle_s.size()
            mixture = mixture.to(self.device)
            oracle_s = oracle_s.to(self.device)
            oracle_ids = oracle_ids.to(self.device)

            net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            assert b == 1, "The batch size of validation dataloader must be 1."
            name = filename

            spk_vectors = net.get_speaker_vectors(mixture)
            b, n_spk, embed_dim, frames = spk_vectors.size()
            spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixture)
            spk_loss, reordered = self.loss_function["spk"](spk_vectors, spk_activity_mask, oracle_ids)
            
            reordered = reordered.mean(-1) # take centroid
            # reordered = self.loss_function["spk"].spk_embeddings[oracle_ids] #.to(self.device)
            # spk_loss = torch.Tensor([0]).to(self.device)

            separated = net.split_waves(mixture, reordered)

            if net.sep_stack.return_all:
                separated = separated[-1]

            sep_loss = self.loss_function["sep"](separated, oracle_s).mean()
            loss = sep_loss + spk_loss

            mixture = mixture.cpu().numpy().squeeze()
            separated = separated.cpu().numpy().squeeze()
            oracle_s = oracle_s.cpu().numpy().squeeze()
            s1_clean = oracle_s[0]
            s2_clean = oracle_s[1]
            
            s1 = separated[0]
            s2 = separated[1]
            s1 = ((s1 - s1.mean())/s1.std())*s1_clean.std()
            s2 = ((s2 - s2.mean())/s2.std())*s2_clean.std()

            # Metrics
            c_m = (compute_SDR(s1_clean, mixture) + compute_SDR(s2_clean, mixture)) / 2
            c_e = (compute_SDR(s1_clean, s1) + compute_SDR(s2_clean, s2)) / 2
            sdr_c_m.append(c_m)
            sdr_c_e.append(c_e)

            # spk distance
            oracle_emds = self.loss_function['spk'].spk_embeddings.gather(dim=0, index=oracle_ids.transpose(1,0).repeat(1,512))

            self.intra_avg_distance.append(torch.norm(reordered[0,:,:]-oracle_emds,dim=1).sum(0).item()/2)
            self.inter_spk_distance.append(torch.norm(reordered[0,0,:]-reordered[0,1,:]).item())
            self.sdr_improve.append(c_e - c_m)

            # visualize spec / audio / spk_id_distribution
            self._visuliza_spec_audio(epoch, i, mixture, s1, s2, s1_clean, s2_clean, name, c_e)

        self.writer.add_scalars(f"Metrics/Validation_SDR", {
            "target and mixture": get_metrics_ave(sdr_c_m),
            "target and enhanced": get_metrics_ave(sdr_c_e)
        }, epoch)
        score = get_metrics_ave(sdr_c_e)
        return score

    @torch.no_grad()
    def _inference(self):
        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)

        sdr_c_m = []  # Clean and mixture
        sdr_c_e = []  # Clean and enhanced

        self.intra_avg_distance = []
        self.inter_spk_distance = []
        self.sdr_improve = []

        for i, (mixture, oracle_s, oracle_ids, filename) in tqdm(enumerate(self.test_dataloader)):
            b, n_spk, frames = oracle_s.size()
            mixture = mixture.to(self.device)
            oracle_s = oracle_s.to(self.device)
            oracle_ids = oracle_ids.to(self.device)

            net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            assert b == 1, "The batch size of validation dataloader must be 1."
            name = filename

            spk_vectors = net.get_speaker_vectors(mixture)
            b, n_spk, embed_dim, frames = spk_vectors.size()

            # clustering using k-means
            reordered = []
            for b in range(spk_vectors.shape[0]):
                cluster_ids, cluster_centers = kmeans(
                    spk_vectors[b].transpose(1, 2).reshape(frames*n_spk, embed_dim),
                    net.n_src,
                    device=spk_vectors.device,
                )
                reordered.append(cluster_centers)

            reordered = torch.stack(reordered)            
            # spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixture)
            # spk_loss, reordered = self.loss_function["spk"](spk_vectors, spk_activity_mask, oracle_ids)
            
            # reordered = reordered.mean(-1) # take centroid
            # reordered = self.loss_function["spk"].spk_embeddings[oracle_ids] #.to(self.device)
            # spk_loss = torch.Tensor([0]).to(self.device)

            separated = net.split_waves(mixture, reordered)

            if net.sep_stack.return_all:
                separated = separated[-1]

            # sep_loss = self.loss_function["sep"](separated, oracle_s).mean()
            # loss = sep_loss + spk_loss

            mixture = mixture.cpu().numpy().squeeze()
            separated = separated.cpu().numpy().squeeze()
            oracle_s = oracle_s.cpu().numpy().squeeze()
            s1_clean = oracle_s[0]
            s2_clean = oracle_s[1]
            
            s1 = separated[0]
            s2 = separated[1]
            s1 = ((s1 - s1.mean())/s1.std())*s1_clean.std()
            s2 = ((s2 - s2.mean())/s2.std())*s2_clean.std()

            # Metrics
            c_m = (compute_SDR(s1_clean, mixture) + compute_SDR(s2_clean, mixture)) / 2
            c_e_p1 = (compute_SDR(s1_clean, s1) + compute_SDR(s2_clean, s2)) / 2
            c_e_p2 = (compute_SDR(s1_clean, s2) + compute_SDR(s2_clean, s1)) / 2
            c_e = max(c_e_p1,c_e_p2)

            sdr_c_m.append(c_m)
            sdr_c_e.append(c_e)

            # spk distance
            # oracle_emds = self.loss_function['spk'].spk_embeddings.gather(dim=0, index=oracle_ids.transpose(1,0).repeat(1,512))

            # self.intra_avg_distance.append(torch.norm(reordered[0,:,:]-oracle_emds,dim=1).sum(0).item()/2)
            self.inter_spk_distance.append(torch.norm(reordered[0,0,:]-reordered[0,1,:]).item())
            self.sdr_improve.append(c_e - c_m)

            # visualize spec / audio / spk_id_distribution
            self._visuliza_spec_audio(0, 0, mixture, s1, s2, s1_clean, s2_clean, name, c_e)

        self.writer.add_scalars(f"Metrics/Test_SDR", {
            "target and mixture": get_metrics_ave(sdr_c_m),
            "target and enhanced": get_metrics_ave(sdr_c_e)
        }, 0)
        score = get_metrics_ave(sdr_c_e)
        return score

    def _visuliza_spec_audio(self, epoch, i, mixture, s1, s2, s1_clean, s2_clean, name, c_e, spk_vct=None, oracle_ids=None):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        sr = self.validation_custom_config["sr"]
            # Visualize audio
        if i <= visualize_audio_limit:
            self.writer.add_audio(f"Speech/{name}_Mixture", mixture, epoch, sample_rate=sr)
            self.writer.add_audio(f"Speech/{name}_s1", s1, epoch, sample_rate=sr)
            self.writer.add_audio(f"Speech/{name}_s2", s2, epoch, sample_rate=sr)
            self.writer.add_audio(f"Speech/{name}_s1_clean", s1_clean, epoch, sample_rate=sr)
            self.writer.add_audio(f"Speech/{name}_s2_clean", s2_clean, epoch, sample_rate=sr)
            # self.writer.add_audio(f"Speech/{name}_Reference", reference, epoch, sample_rate=sr)

        # Visualize waveform
        if i <= visualize_waveform_limit:
            fig, ax = plt.subplots(5, 1)
            for j, y in enumerate([mixture, s1, s1_clean, s2, s2_clean]):
                ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                    np.mean(y),
                    np.std(y),
                    np.max(y),
                    np.min(y)
                ))
                librosa.display.waveshow(y, sr=sr, ax=ax[j])
            plt.tight_layout()
            self.writer.add_figure(f"Waveform/{name}", fig, epoch)

        # Visualize spectrogram
        mixture_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160))
        s1_mag, _ = librosa.magphase(librosa.stft(s1, n_fft=320, hop_length=160))
        s2_mag, _ = librosa.magphase(librosa.stft(s2, n_fft=320, hop_length=160))
        s1_clean_mag, _ = librosa.magphase(librosa.stft(s1_clean, n_fft=320, hop_length=160))
        s2_clean_mag, _ = librosa.magphase(librosa.stft(s2_clean, n_fft=320, hop_length=160))

        # print(f"Value: {c_e - c_m} \n"
        #       f"Mean: {get_metrics_ave(sdr_c_e) - get_metrics_ave(sdr_c_m)}")

        # visulize the speaker id
        # sizes = 10
        fig, ax = plt.subplots()
        if self.intra_avg_distance:
            ax.scatter(self.intra_avg_distance, self.sdr_improve, color='tab:blue',alpha=0.5)
        ax.scatter(self.inter_spk_distance, self.sdr_improve, color='tab:red',alpha=0.5)
        # plt.axis('equal')
        ax.set_xlim(0,2)
        ax.set_ylim(-30,30)
        self.writer.add_figure("spk_distance/sdr", fig, epoch)

        if i <= visualize_spectrogram_limit:
            fig, axes = plt.subplots(5, 1, figsize=(6, 10))
            for k, mag in enumerate([
                mixture_mag,
                s1_mag,
                s1_clean_mag,
                s2_mag,
                s2_clean_mag
            ]):
                axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                f"std: {np.std(mag):.3f}, "
                                #   f"max: {np.max(mag):.3f}, "
                                #   f"min: {np.min(mag):.3f}, "
                                f"sisdr: {c_e:.3f}"
                                )
                librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k],
                                        sr=sr)
            plt.tight_layout()
            self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <root_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
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

        # spk embedding table in spk stack loss
        state_dict["loss_spk_emd_table"] = self.loss_function['spk'].state_dict()
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

        if self.load_spk_emd:
            self.loss_function['spk'].load_state_dict(checkpoint["loss_spk_emd_table"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

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
            spk_emd_table = model_checkpoint['loss_spk_emd_table']
            model_checkpoint = model_checkpoint['model']
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_checkpoint)
        else:
            self.model.load_state_dict(model_checkpoint)
        
        if self.load_spk_emd:
            self.loss_function['spk'].load_state_dict(spk_emd_table)

        print(f"Model preloaded successfully from {model_path.as_posix()}.")
