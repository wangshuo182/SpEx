import torch
from torch import nn
from asteroid.losses.sdr import MultiSrcNegSDR
from itertools import permutations
import math
from torch.nn import functional as F
import numpy as np


class ClippedSDR(nn.Module):
    def __init__(self, clip_value=-30):
        super(ClippedSDR, self).__init__()

        self.snr = MultiSrcNegSDR("sisdr")
        self.clip_value = float(clip_value)

    def forward(self, est_targets, targets):

        return torch.clamp(self.snr(est_targets, targets), min=self.clip_value)

class SpeakerVectorLoss(nn.Module):
    def __init__(
        self,
        n_speakers,
        embed_dim=512,
        learnable_emb=False,
        loss_type="global",
        weight=2,
        distance_reg=0.0,
        gaussian_reg=0.0,
        return_oracle=False,
    ):
        super(SpeakerVectorLoss, self).__init__()

        self.learnable_emb = learnable_emb
        assert loss_type in ["distance", "global", "local"]
        self.loss_type = loss_type
        self.weight = float(weight)
        self.distance_reg = float(distance_reg)
        self.gaussian_reg = float(gaussian_reg)
        self.return_oracle = return_oracle

        spk_emb = torch.eye(max(n_speakers, embed_dim))  # Neil: one-hot init
        spk_emb = spk_emb[:n_speakers, :embed_dim]

        if learnable_emb == True:
            self.spk_embeddings = nn.Parameter(spk_emb)
        else:
            self.register_buffer("spk_embeddings", spk_emb)

        if self.loss_type != "distance":
            self.alpha = nn.Parameter(torch.Tensor([1.0]))
            self.beta = nn.Parameter(torch.Tensor([0.0]))

    @staticmethod
    def _l_dist_speaker(c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        spk_labels = spk_labels.unsqueeze(-1).repeat(1, 1, spk_embeddings.size(-1))
        utt_embeddings = spk_embeddings.unsqueeze(0).repeat(spk_labels.size(0), 1, 1).gather(
            1, spk_labels
        ).unsqueeze(-1) * spk_mask.unsqueeze(2)

        distance = ((c_spk_vec_perm - utt_embeddings) ** 2).sum(2)

        intra_spk = ((c_spk_vec_perm[:, 0] - c_spk_vec_perm[:, 1]) ** 2).sum(1)
        return distance.sum(1) + F.relu(1.0 - intra_spk) * 2  # ok for two speakers

    def _l_local_speaker(self, c_spk_vec_perm, spk_embeddings, spk_labels, spk_mask):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
        alpha = torch.clamp(self.alpha, 1e-8)

        distance = alpha * ((c_spk_vec_perm - utt_embeddings) ** 2).sum(2) + self.beta
        distances = (
            alpha * ((c_spk_vec_perm.unsqueeze(1) - utt_embeddings.unsqueeze(2)) ** 2).sum(3)
            + self.beta
        )

        # return (distance + torch.log(torch.exp(-distances).sum(1))).sum(1)
        return (distance + torch.logsumexp(-distances, dim=1)).sum(1)
        


    def _l_global_speaker(self, c_spk_vec_perm, spk_embeddings, spk_labels):

        utt_embeddings = spk_embeddings[spk_labels].unsqueeze(-1)
        alpha = torch.clamp(self.alpha, min=1e-8)

        distance_utt = alpha * ((c_spk_vec_perm - utt_embeddings) ** 2).sum(2) + self.beta

        B, src, embed_dim, frames = c_spk_vec_perm.size()
        spk_embeddings = spk_embeddings.reshape(1, spk_embeddings.shape[0], embed_dim, 1).expand(
            B, -1, -1, frames
        )
        distances = (
            alpha * ((c_spk_vec_perm.unsqueeze(1) - spk_embeddings.unsqueeze(2)) ** 2).sum(3) + self.beta
        )
        return (distance_utt + torch.logsumexp(-distances, dim=1)).sum(1)
        # return (distance_utt + torch.log(torch.exp(-distances).sum(1))).sum(1)

        # exp normalize trick
        # with torch.no_grad():
        #   b = torch.max(distances, dim=1, keepdim=True)[0]
        # out = -distance_utt + b.squeeze(1) - torch.log(torch.exp(-distances + b).sum(1))
        # return out.sum(1)

    def forward(self, speaker_vectors, spk_mask, spk_labels):

        if self.gaussian_reg:
            noise = torch.randn(
                self.spk_embeddings.size(), device=speaker_vectors.device
            ) * math.sqrt(self.gaussian_reg)
            self.spk_embeddings = self.spk_embeddings + noise.to(spk_labels)
        # else:
        #     spk_embeddings = self.spk_embeddings

        if self.learnable_emb or self.gaussian_reg:  # re project on unit sphere
            # re-project on unit sphere
            self.spk_embeddings = nn.Parameter(nn.functional.normalize(self.spk_embeddings))
            # self.spk_embeddings = (
            #     self.spk_embeddings / torch.sum(self.spk_embeddings ** 2, -1, keepdim=True).sqrt()
            # )

        if self.distance_reg:
            pairwise_dist = (
                (torch.abs(self.spk_embeddings.unsqueeze(0) - self.spk_embeddings.unsqueeze(1)))
                .mean(-1)
                .fill_diagonal_(np.inf)
            )
            distance_reg = -torch.sum(torch.min(torch.log(pairwise_dist), dim=-1)[0])

        # speaker vectors B, n_src, dim, frames
        # spk mask B, n_src, frames boolean mask
        # spk indxs list of len B of list which contains spk label for current utterance
        B, n_src, embed_dim, frames = speaker_vectors.size()

        perms = list(permutations(range(n_src)))
        if self.loss_type == "distance":
            loss_set = torch.stack(
                [
                    self._l_dist_speaker(
                        speaker_vectors[:, perm], self.spk_embeddings, spk_labels, spk_mask
                    )
                    for perm in perms
                ],
                dim=1,
            )
        elif self.loss_type == "local":
            loss_set = torch.stack(
                [
                    self._l_local_speaker(
                        speaker_vectors[:, perm], self.spk_embeddings, spk_labels, spk_mask
                    )
                    for perm in perms
                ],
                dim=1,
            )
        else:
            loss_set = torch.stack(
                [
                    self._l_global_speaker(
                        speaker_vectors[:, perm], self.spk_embeddings, spk_labels
                    )
                    for perm in perms
                ],
                dim=1,
            )
            
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # reorder sources for each frame !!
        perms = min_loss.new_tensor(perms, dtype=torch.long)
        perms = perms[..., None, None].expand(-1, -1, B, frames)
        min_loss_idx = min_loss_idx[None, None, ...].expand(1, n_src, -1, -1)
        min_loss_perm = torch.gather(perms, dim=0, index=min_loss_idx)[0]
        min_loss_perm = (
            min_loss_perm.transpose(0, 1).reshape(B, n_src, 1, frames).expand(-1, -1, embed_dim, -1)
        )
        # tot_loss

        spk_loss = self.weight * min_loss.mean()
        if self.distance_reg:
            spk_loss += self.distance_reg * distance_reg
        reordered_sources = torch.gather(speaker_vectors, dim=1, index=min_loss_perm)

        if self.return_oracle:
            self.spk_embeddings = self.spk_embeddings.to(spk_labels)
            utt_embeddings = self.spk_embeddings[spk_labels].unsqueeze(-1) * spk_mask.unsqueeze(2)
            return spk_loss, reordered_sources, utt_embeddings

        return spk_loss, reordered_sources

def si_sdr_loss():
    def _loss_function(ref, est):
        ref = ref - torch.mean(ref)
        est = est - torch.mean(est)
        reference_energy = torch.sum(ref ** 2, dim=-1, keepdim=True)
        optimal_scaling = torch.sum(ref * est, dim=-1, keepdim=True) / reference_energy
        projection = optimal_scaling * ref
        noise = est - projection
        ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
        # return 10 * np.log10(ratio)
        return -torch.mean(10 * torch.log10(ratio))

    return _loss_function

def multi_scale_si_sdr_loss(weights):
    def _loss_function(ref, short_est, middle_est, long_est):
        assert len(weights) == 3
        short_si_sdr_loss = si_sdr_loss()(ref, short_est)
        middle_si_sdr_loss = si_sdr_loss()(ref, middle_est)
        long_si_sdr_loss = si_sdr_loss()(ref, long_est)
        loss = weights[0] * short_si_sdr_loss + weights[1] * middle_si_sdr_loss + weights[2] * long_si_sdr_loss
        return loss, (short_si_sdr_loss, middle_si_sdr_loss, long_si_sdr_loss)

    return _loss_function

def wavesplit_loss(n_speakers=2, embed_dim=512, loss_type="distance", gaussian_reg=0, distance_reg=0, weight=2, learnable_emb=False):
    loss_spk = SpeakerVectorLoss(
        n_speakers, embed_dim, loss_type=loss_type, gaussian_reg=gaussian_reg, distance_reg=distance_reg, weight=weight, learnable_emb=learnable_emb
    )
    loss_sep = ClippedSDR()
    return {'spk':loss_spk, 'sep':loss_sep}

if __name__ == '__main__':
    # ClipedSDR
    est = torch.rand((5, 1, 16384))
    ref = torch.rand((5, 1, 16384))
    print(wavesplit_loss(n_speakers=2, embed_dim=512)['sep'](est,ref))

    # spk

    import random

    n_speakers = 101
    emb_speaker = 256

    spk_embedding_1 = torch.rand(emb_speaker) * 0.01  # torch.zeros(emb_speaker).float()
    spk_embedding_2 = torch.rand(emb_speaker) * 0.01  # torch.zeros(emb_speaker).float()
    spk_embedding_1[0] = 1.0
    spk_embedding_2[1] = 1.0

    tmp = []
    for i in range(200):
        if np.random.random() >= 0.5:
            tmp.append(torch.stack((spk_embedding_1, spk_embedding_2)))
        else:
            tmp.append(torch.stack((spk_embedding_2, spk_embedding_1)))

    speaker_vectors = torch.stack(tmp).permute(1, 2, 0).unsqueeze(0)

    speaker_labels = torch.from_numpy(np.array([[1, 0]]))
    oracle = (
        torch.stack((spk_embedding_2, spk_embedding_1))
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(1, 1, 1, 200)
    )

    # testing exp normalize average
    # distances = torch.ones((1, 101, 4000))
    # with torch.no_grad():
    #   b = torch.max(distances, dim=1, keepdim=True)[0]
    # out = b.squeeze(1) - torch.log(torch.exp(-distances + b).sum(1))
    # out2 = - torch.log(torch.exp(-distances).sum(1))

    loss_spk = SpeakerVectorLoss(
        n_speakers, emb_speaker, loss_type="distance", distance_reg=0, gaussian_reg=0
    )

    speaker_mask = torch.ones(
        (1, 2, 200)
    )  # silence where there are no speakers actually thi is test
    spk_loss, reordered = loss_spk(speaker_vectors, speaker_mask, speaker_labels)
    print(spk_loss)
    np.testing.assert_array_almost_equal(reordered.numpy(), oracle.numpy())

    # SpEx SDR
    print(multi_scale_si_sdr_loss([0.1, 0.2, 0.7])(est, ref, ref, ref))
