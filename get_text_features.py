import pandas as pd
from tqdm.auto import tqdm

import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def get_sentence_embeddings():
    df = pd.read_csv("data/combine_ds_geode_all.csv", usecols=['blip_cap_without_topic','blip_cap_with_topic','ram_cap_with_tags','ram_cap_without_tags'])
    # images = list(df['image'])
    captions_1 = list(df['blip_cap_without_topic'])
    captions_2 = list(df['blip_cap_with_topic'])
    captions_3 = list(df['ram_cap_with_tags'])
    captions_4 = list(df['ram_cap_without_tags'])
    captions_all = [[captions_1[i], captions_2[i], captions_3[i], captions_4[i]] for i in range(len(captions_1))]

    SBert_caption1 = model.encode(captions_1, show_progress_bar=True)
    SBert_caption2 = model.encode(captions_2, show_progress_bar=True)
    SBert_caption3 = model.encode(captions_3, show_progress_bar=True)
    SBert_caption4 = model.encode(captions_4, show_progress_bar=True)
    SBert_captions_all = model.encode(captions_all, show_progress_bar=True)
    np.save("data/features/caption1_SBERT.npy", SBert_caption1)
    np.save("data/features/caption2_SBERT.npy", SBert_caption2)
    np.save("data/features/caption3_SBERT.npy", SBert_caption3)
    np.save("data/features/caption4_SBERT.npy", SBert_caption4)
    np.save("data/features/caption_all_SBERT.npy", SBert_captions_all)




if __name__ == '__main__':
    get_sentence_embeddings()