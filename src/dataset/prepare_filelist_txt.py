import os

def save_wav_path_to_txt(root_path, split, pairs):
    file_list = os.listdir(os.path.join(root_path, split, pairs[0]))

    dst = root_path + '/' + 'wav_list_' + split + '.txt'
    text_line_list = []
    for filename in file_list:
        for pair in pairs[1:]:
            assert os.path.exists(os.path.join(root_path, split, pair, filename))
        text_line_list.append(' '.join(['/'.join([root_path, split, i, filename]) for i in pairs]) + '\n')

    f = open(dst, "w+")
    for text in text_line_list:
        f.write(text)
    f.close
    print(str(split)+' '+"Done!")

if __name__ == "__main__":
    # root_path = '/media/asus/DATADISK/DATASETS/wsj0-mix/2speakers/wav8k/min'
    # root_path = '/workspace/myDataset/wsj0-mix/2speakers/wav8k/min'
    root_path = '/mnt/default/data/wsj0-mix/2speakers/wav8k/min'
    splits = ['tr', 'cv', 'tt']
    pairs = ['mix', 's1', 's2']
    # pairs = ['mix', 's1', 'aux']
    print(f"preparing dataset list file in <{root_path}>")

    for split in splits:
        save_wav_path_to_txt(root_path, split, pairs)
