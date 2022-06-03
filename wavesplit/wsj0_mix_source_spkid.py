import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf

# # WHAM tasks
# enh_single = {"mixture": "mix_single", "sources": ["s1"], "infos": ["noise"], "default_nsrc": 1}
# enh_both = {"mixture": "mix_both", "sources": ["mix_clean"], "infos": ["noise"], "default_nsrc": 1}
# sep_clean = {"mixture": "mix_clean", "sources": ["s1", "s2"], "infos": [], "default_nsrc": 2}
# sep_noisy = {"mixture": "mix_both", "sources": ["s1", "s2"], "infos": ["noise"], "default_nsrc": 2}

# WHAM_TASKS = {
#     "enhance_single": enh_single,
#     "enhance_both": enh_both,
#     "sep_clean": sep_clean,
#     "sep_noisy": sep_noisy,
# }
# # Aliases.
# WHAM_TASKS["enh_single"] = WHAM_TASKS["enhance_single"]
# WHAM_TASKS["enh_both"] = WHAM_TASKS["enhance_both"]


class Dataset(data.Dataset):
    """Dataset class for wsj0-2mix source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
    """

    def __init__(self, dataset_list, limit=None, offset=0, sample_rate=8000, segment=None, nondefault_nsrc=2):
        super(Dataset, self).__init__()

        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset_list)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.sample_rate = sample_rate
        self.seg_len = None if segment is None else int(segment * sample_rate)

        self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json examples
        # ex_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")

        # with open(ex_json, "r") as f:
        #     examples = json.load(f)

        # Filter out short utterances only when segment is specified
        # self.examples = []
        # orig_len = len(examples)
        # drop_utt, drop_len = 0, 0
        # if not self.like_test:
        #     for ex in examples:  # Go backward
        #         if ex["length"] < self.seg_len:
        #             drop_utt += 1
        #             drop_len += ex["length"]
        #         else:
        #             self.examples.append(ex)

        # print(
        #     "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
        #         drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
        #     )
        # )

        # count total number of speakers
        mix_wav_path_list = [item.split(" ")[0] for item in dataset_list]
        self.speaker_dict = self.build_speaker_dict(mix_wav_path_list)
        self.speaker_id_list = list(self.speaker_dict.keys())
        print("Number of the speakers is: ", len(self.speaker_id_list))

        # speakers = set()
        # for ex in self.dataset_list:
        #     for spk in ex["spk_id"]:
        #         speakers.add(spk[:3])

        # print("Total number of speakers {}".format(len(list(speakers))))

        # convert speakers id into integers
        # mapping spkid to int sequence
        indx = 0
        spk2indx = {}
        for spk in self.speaker_id_list:
            spk2indx[spk] = indx
            indx += 1
        self.spk2indx = spk2indx

        self.mix_wav_spk_list = []
        for ex in mix_wav_path_list:
            tmp = []
            spkids = self.get_speaker_id(self.get_filename(ex))
            for spk in spkids:
                tmp.append(spk2indx[spk])
            self.mix_wav_spk_list.append(tmp)


    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        mix_s1_s2_path = self.dataset_list[idx]
        mix_path, s1_path, s2_path = mix_s1_s2_path.split(' ')

        # Load mixture
        x, _ = sf.read(mix_path, dtype="float32")
        sample_len = len(x)

        # Random start
        if sample_len <= self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, sample_len - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len

        x = x[rand_start:stop]
        s1, _ = sf.read(s1_path, start=rand_start, stop=stop, dtype="float32")
        s2, _ = sf.read(s2_path, start=rand_start, stop=stop, dtype="float32")
        x = self.sample_fixed_length_padding(x, self.seg_len)
        s1 = self.sample_fixed_length_padding(s1, self.seg_len)
        s2 = self.sample_fixed_length_padding(s2, self.seg_len)

        assert len(x) == len(s1) == len(s2) == self.seg_len
        source_arrays = [s1, s2]

        # source_arrays = []
        # for src in c_ex["sources"]:
        #     s, _ = sf.read(src, start=rand_start, stop=stop, dtype="float32")
        #     source_arrays.append(s)

        sources = torch.from_numpy(np.vstack(source_arrays))
        spkid = [self.mix_wav_spk_list[idx][0], self.mix_wav_spk_list[idx][1]]

        if np.random.random() > 0.5:  # randomly permute (not sure if it can help but makes sense)
            sources = torch.stack((sources[1], sources[0]))
            spkid = [self.mix_wav_spk_list[idx][1], self.mix_wav_spk_list[idx][0]]
            # c_ex["spk_id"] = [c_ex["spk_id"][1], c_ex["spk_id"][0]]

        return torch.from_numpy(x), sources, torch.Tensor(spkid).long()

    def build_speaker_dict(self, wav_path_list):
        """

        """
        # random.shuffle(wav_path_list)

        fully_speaker_dict = {}
        for file_path in wav_path_list:
            speaker_id = self.get_speaker_id(self.get_filename(file_path))

            if speaker_id[0] in fully_speaker_dict:
                fully_speaker_dict[speaker_id[0]].append(file_path)
            else:
                fully_speaker_dict[speaker_id[0]] = [file_path]

            if speaker_id[1] in fully_speaker_dict:
                fully_speaker_dict[speaker_id[1]].append(file_path)
            else:
                fully_speaker_dict[speaker_id[1]] = [file_path]

        return fully_speaker_dict

    @staticmethod
    def get_speaker_id(filename):
        '''
        filename format:
        443c020i_0.75055_442o030n_-0.75055.wav
        speaker 1: 443
        speaker 2: 442
        '''
        speaker_id = [filename.split("_")[0][:3], filename.split("_")[2][:3]]
        return speaker_id

    @staticmethod
    def get_filename(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return filename

    @staticmethod
    def sample_fixed_length_padding(data, sample_length):
        """
        sample with fixed length from two dataset
        """
        # noise_eps = 10E-8 * np.random.randn(sample_length)
        if len(data) == sample_length:
            return data
        else:
            frames_total = len(data)
            return np.append(data, 10E-10*np.random.randn(sample_length - frames_total).astype('float32'))

if __name__ == "__main__":
    a = Dataset(
        "/media/asus/DATADISK/DATASETS/wsj0-mix/2speakers/wav8k/min/wav_list_tr.txt"
    )

    for i in a:
        print(i[-1])
