import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
from PIL import Image
import numpy as np
from transformers import pipeline, BlipForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from main import path_data_geode, path_images_geode, index_geode, get_metadata

model = SentenceTransformer('all-mpnet-base-v2')


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def img_captioning():
    objects, _, _, images = get_metadata()

    #TODO: CONDITIONAL GENERATION: https://huggingface.co/docs/transformers/model_doc/blip
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    text = "A picture of [object name]" # TODO
    # image = Image.open(image_names)
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model(**inputs)


    image_to_text1 = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # image_to_text1 = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    # image_to_text2 = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b")

    image_names = ["/".join([path_images_geode, p]) for p in index_geode['file_path']]
    list_captions1, list_captions2 = [], []

    dataset = ListDataset(image_names)
    for out in tqdm(image_to_text1(dataset)):
        list_captions1.append(out[0]['generated_text'])
    # list_captions1 += ['lala'] * (len(index['file_path']) - 2)

    df = pd.read_csv(path_data_geode + "/feat_CLIP.csv")
    df["Caption1"] = list_captions1
    df.to_csv(path_data_geode + "/feat_all.csv")

    # for out in tqdm(image_to_text2(dataset)):
    #     list_captions2.append(out[0]['generated_text'])

    # df["Caption2"] = list_captions2
    # df.to_csv(path_data_geode + "/feat_all.csv")


def get_sentence_embeddings():
    df = pd.read_csv(path_data_geode + "/feat_all.csv")
    captions1 = list(df['Caption1'])
    # captions2 = list(df['Caption2'])
    SBert_caption1 = model.encode(captions1, show_progress_bar=True)
    # SBert_caption2 = model.encode(captions2)
    # df["Caption1_SBert"] = SBert_caption1
    # df["Caption2_SBert"] = SBert_caption2
    # df.to_csv(path_data_geode + "/feat_all.csv")
    np.save(path_data_geode + "/caption1_SBERT.npy", SBert_caption1)
    # np.save(path_data_geode + "/caption2_SBERT.npy", SBert_caption2)


if __name__ == '__main__':
    print("sth")
    # img_captioning()
    # get_sentence_embeddings()