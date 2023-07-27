from collections import Counter

import numpy as np
import pandas as pd

df_all = pd.read_csv("data/combine_ds_geode_all.csv")

def get_metadata():
    # threshold = 1000
    objects = list(df_all['topic'])
    countries = list(df_all['country'])
    regions = list(df_all['region'])
    continents = list(df_all['continent'])
    income = list(df_all['quartile'])
    images = list(df_all['image'])

    return objects, countries, regions, continents, income, images

def filter_data(objects, countries, regions, continents, income, images):
    #TODO
    # remove data with few datapoints
    # countries_to_remove = [country[0] for country in Counter(countries).items() if country[1] < threshold]
    # indexes = [i for i, x in enumerate(countries) if x not in countries_to_remove]
    indexes = [i for i, x in enumerate(countries)]

    objects_filtered = [objects[i] for i in indexes]
    countries_filtered = [countries[i] for i in indexes]
    regions_filtered = [regions[i] for i in indexes]
    continents_filtered = [continents[i] for i in indexes]
    incomes_filtered = [income[i] for i in indexes]
    images_filtered = [images[i] for i in indexes]

def get_features():
    '''
        CLIP features
    '''
    img_CLIP_feat = np.load("data/features/image_CLIP.npy")
    txt_CLIP_feat = np.load("data/features/txt_CLIP.npy")

    '''
        SBert features from image captions
    '''
    caption1_SBERT_feat = np.load("data/features/caption1_SBERT.npy")
    caption2_SBERT_feat = np.load("data/features/caption2_SBERT.npy")
    caption3_SBERT_feat = np.load("data/features/caption3_SBERT.npy")
    caption4_SBERT_feat = np.load("data/features/caption4_SBERT.npy")
    caption_all_SBERT_feat = np.load("data/features/caption_all_SBERT.npy")

    # print(caption1_SBERT_feat.shape, caption_all_SBERT_feat.shape, img_CLIP_feat.shape, txt_CLIP_feat.shape)
    return img_CLIP_feat, txt_CLIP_feat, caption1_SBERT_feat, caption2_SBERT_feat, caption3_SBERT_feat, caption4_SBERT_feat, caption_all_SBERT_feat



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_metadata()
    get_features()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
