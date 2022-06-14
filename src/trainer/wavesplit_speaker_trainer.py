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

        for iter, (mixture, oracle_s, oracle_ids, _) in enumerate(tqdm(self.train_dataloader, desc="Training")):
            mixture = mixture.to(self.device)
            oracle_ids = oracle_ids.to(self.device)

            self.optimizer.zero_grad()
            net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

            spk_vectors = net.get_speaker_vectors(mixture)
            b, n_spk, _, frames = spk_vectors.size()
            spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixture)
            spk_loss, _ = self.loss_function["spk"](spk_vectors, spk_activity_mask, oracle_ids)
            
            spk_loss.backward()
            all_parameters = list(net.parameters()) + list(self.loss_function["spk"].parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(all_parameters, 2)
            self.optimizer.step()

            self.writer.add_scalar("train/spk_loss", spk_loss, iter + (epoch-1)*len(self.train_dataloader))
            self.writer.add_scalar("train/epoch", epoch, iter + (epoch-1)*len(self.train_dataloader))

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)

        samples = {'name':[], 'intra_avg_distance':[], 'inter_spk_distance':[], 'sdr_improve':[]}
        loss_sum = 0

        for i, (mixture, oracle_s, oracle_ids, filename) in tqdm(enumerate(self.validation_dataloader)):
            b, n_spk, frames = oracle_s.size()
            assert b == 1, "The batch size of validation dataloader must be 1."
            mixture = mixture.to(self.device)
            oracle_ids = oracle_ids.to(self.device)
            net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

            spk_vectors = net.get_speaker_vectors(mixture)
            b, n_spk, embed_dim, frames = spk_vectors.size()
            spk_activity_mask = torch.ones((b, n_spk, frames)).to(mixture)
            spk_loss, reordered = self.loss_function["spk"](spk_vectors, spk_activity_mask, oracle_ids)

            loss_sum += spk_loss
            reordered = reordered.mean(-1) # take centroid

            mixture = mixture.cpu().numpy().squeeze()
            oracle_s = oracle_s.cpu().numpy().squeeze()
            s1_clean = oracle_s[0]
            s2_clean = oracle_s[1]

            # spk distance
            oracle_emds = self.loss_function['spk'].spk_embeddings.gather(dim=0, index=oracle_ids.transpose(1,0).repeat(1,embed_dim))
            samples['intra_avg_distance'].append(torch.norm(reordered[0,:,:]-oracle_emds,dim=1).sum(0).item()/2)
            samples['inter_spk_distance'].append(torch.norm(reordered[0,0,:]-reordered[0,1,:]).item())
            samples['name'].append(filename)
        
        # Loss
        self.writer.add_scalar("validation/spk_loss", loss_sum/len(self.validation_dataloader), epoch)
        # intra_spk_distance / inter_spk_distance / cross spk distance
        self._visuliza_spk_distribution(samples, epoch, spk_emd_table=self.loss_function['spk'].spk_embeddings, spk2indx=self.train_dataloader.dataset.spk2indx, prefix='val')
        return loss_sum/len(self.validation_dataloader)

    @torch.no_grad()
    def _inference(self, epoch=0):
        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)

        samples = {
            'name':[],
            'intra_avg_distance':[],
            'inter_spk_distance':[],
            'sdr_improve':[]
        }   
        for i, (mixture, oracle_s, oracle_ids, filename) in tqdm(enumerate(self.test_dataloader)):
            b, n_spk, frames = oracle_s.size()
            assert b == 1, "The batch size of validation dataloader must be 1."
            mixture = mixture.to(self.device)
            oracle_ids = oracle_ids.to(self.device)
            net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

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
            # spk distance
            # oracle_emds = self.loss_function['spk'].spk_embeddings.gather(dim=0, index=oracle_ids.transpose(1,0).repeat(1,embed_dim))
            # samples['intra_avg_distance'].append(torch.norm(reordered[0,:,:]-oracle_emds,dim=1).sum(0).item()/2)
            samples['inter_spk_distance'].append(torch.norm(reordered[0,0,:]-reordered[0,1,:]).item())
            samples['name'].append(filename)

        # intra_spk_distance / inter_spk_distance+

        self._visuliza_spk_distribution(samples, epoch, prefix='test')
        # cross spk distance

        return get_metrics_ave(samples['inter_spk_distance'])

    def _visuliza_spec_audio(self, epoch, i, mixture, s1, s2, s1_clean, s2_clean, name, c_e, spk_vct=None, oracle_ids=None, prefix='_'):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        sr = self.validation_custom_config["sr"]
            # Visualize audio
        if i <= visualize_audio_limit:
            self.writer.add_audio(f"{prefix}Speech/{name}_Mixture", mixture, epoch, sample_rate=sr)
            self.writer.add_audio(f"{prefix}Speech/{name}_s1", s1, epoch, sample_rate=sr)
            self.writer.add_audio(f"{prefix}Speech/{name}_s2", s2, epoch, sample_rate=sr)
            self.writer.add_audio(f"{prefix}Speech/{name}_s1_clean", s1_clean, epoch, sample_rate=sr)
            self.writer.add_audio(f"{prefix}Speech/{name}_s2_clean", s2_clean, epoch, sample_rate=sr)
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
            self.writer.add_figure(f"{prefix}Waveform/{name}", fig, epoch)

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
        self.writer.add_figure(f"{prefix}spk_distance/sdr", fig, epoch)

        # Visualize spectrogram
        if i <= visualize_spectrogram_limit:
            mixture_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160))
            s1_mag, _ = librosa.magphase(librosa.stft(s1, n_fft=320, hop_length=160))
            s2_mag, _ = librosa.magphase(librosa.stft(s2, n_fft=320, hop_length=160))
            s1_clean_mag, _ = librosa.magphase(librosa.stft(s1_clean, n_fft=320, hop_length=160))
            s2_clean_mag, _ = librosa.magphase(librosa.stft(s2_clean, n_fft=320, hop_length=160))
            
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
            self.writer.add_figure(f"{prefix}Spectrogram/{name}", fig, epoch)

    def _visuliza_spk_distribution(self, samples, epoch, spk_emd_table=None, spk2indx=None, prefix='_'):
        # visulize the speaker distance
        
        if samples['sdr_improve']:
            fig, ax = plt.subplots()
            ax.scatter(samples['inter_spk_distance'], samples['sdr_improve'], color='tab:red',alpha=0.5)
            if samples['intra_avg_distance']:
                ax.scatter(samples['intra_avg_distance'], samples['sdr_improve'], color='tab:blue',alpha=0.5)
            ax.set_xlim(0,2)
            ax.set_ylim(-30,30)
            self.writer.add_figure(f"{prefix}spk_distance/sdr", fig, epoch)

        # visulize the distance distribution
        num_bins = 30
        fig, ax = plt.subplots()
        # the histogram of the data
        n, bins, patches = ax.hist(samples['inter_spk_distance'], num_bins, color='tab:red', alpha=0.5)
        ax.set_xlabel('distance')
        ax.set_ylabel('sample times')
        # ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
        self.writer.add_figure(f"{prefix}spk_distance/inter", fig, epoch)
        fig, ax = plt.subplots()
        if samples['intra_avg_distance']:
            n, bins, patches = ax.hist(samples['intra_avg_distance'], num_bins, color='tab:blue', alpha=0.5)
            self.writer.add_figure(f"{prefix}spk_distance/intra", fig, epoch)

        # spk embedding table
        if spk_emd_table is not None:
            spk_cross_distance = (spk_emd_table @ spk_emd_table.T).cpu().numpy()
            fig = plt.figure(figsize=(10,10))
            ax = fig.subplots()
            ax.imshow(spk_cross_distance)

            distance_mean = spk_cross_distance.mean()
            distance_max = spk_cross_distance.max()

            ax.set_title(f"max:{distance_max:.2f}, mean:{distance_mean:.3f}")

            # indx2spk = dict([val,key] for key,val in spk2indx.items())
            # spk_list = [indx2spk[i] for i in range(len(indx2spk))]
            # ax.set_xticks(np.arange(len(spk_list)), labels=spk_list)
            # ax.set_yticks(np.arange(len(spk_list)), labels=spk_list)
            # plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

            # for i in range(len(spk_list)):
            #     for j in range(len(spk_list)):
            #         text = ax.text(j, i, spk_cross_distance[i, j], ha="center", va="center", color="w", fontsize=4)
            self.writer.add_figure(f"{prefix}spk_cross_distance", fig, epoch)

    @torch.no_grad()
    def _test_epoch(self, epoch):
        return self._inference(epoch=epoch)
        
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
            "optimizer": self.optimizer.state_dict(),
            "model": {
                "spk": None,
                "sep": None
            }
        }

        net = self.model.module.cpu() if isinstance(self.model, torch.nn.DataParallel) else self.model.cpu()
        # speaker / separation / emd table
        state_dict["model"]["spk"] = net.spk_stack.state_dict()
        state_dict["model"]["emd"] = self.loss_function['spk'].state_dict()
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

        net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        net.spk_stack.load_state_dict(checkpoint["model"]['spk'])
        # net.sep_stack.load_state_dict(checkpoint["model"]['sep'])
        self.loss_function['spk'].load_state_dict(checkpoint["model"]['emd'])

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
            model_checkpoint = model_checkpoint['model']

        net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if self.load_spk_net: net.spk_stack.load_state_dict(model_checkpoint['spk'])
        if self.load_sep_net: net.sep_stack.load_state_dict(model_checkpoint['sep'])
        if self.load_spk_emd: self.loss_function['spk'].load_state_dict(model_checkpoint['emd'])

        print(f"Model preloaded successfully from {model_path.as_posix()}.")
