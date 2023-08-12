from collections import Counter

import numpy as np
import pandas as pd

df_all = pd.read_csv("data/combine_ds_geode_all.csv")


def get_metadata():
    objects = list(df_all['topic'])
    countries = list(df_all['country'])
    regions = list(df_all['region'])
    continents = list(df_all['continent'])
    income = list(df_all['quartile'])
    images = list(df_all['image'])

    return objects, countries, regions, continents, income, images


# remove data with few datapoints: objects & countries with few data
def filter_data(objects, countries, regions, continents, income, images):
    '''
    Filter out objects with less than 1000 images
    Filter out countries with less than 100 images
    '''
    threshold_object = 100
    objects_to_remove = [object_[0] for object_ in Counter(objects).items() if object_[1] < threshold_object]
    print(f"Removing {len(objects_to_remove)} from {len(set(objects))}, new #total objects: {len(set(objects)) - len(objects_to_remove)}")

    threshold_country = 100
    countries_to_remove = [country[0] for country in Counter(countries).items() if country[1] < threshold_country]
    print(f"Removing {len(countries_to_remove)} from {len(set(countries))}, new #total countries: {len(set(countries)) - len(countries_to_remove)}")

    indexes = [i for i, (obj, country) in enumerate(zip(objects, countries))
               if obj not in objects_to_remove and country not in countries_to_remove]

    # indexes_removed = [i for i in range(len(objects)) if i not in indexes]
    #
    # objects_filtered = [objects[i] for i in indexes]
    # countries_filtered = [countries[i] for i in indexes]
    # regions_filtered = [regions[i] for i in indexes]
    # continents_filtered = [continents[i] for i in indexes]
    # incomes_filtered = [income[i] for i in indexes]
    # images_filtered = [images[i] for i in indexes]
    #
    # print(Counter(objects_filtered))
    # print(Counter(countries_filtered))
    # print(Counter(regions_filtered))
    # print(Counter(continents_filtered))
    # print(len(set(images_filtered)), len(set(objects_filtered)), len(set(countries_filtered)),
    #           len(set(regions_filtered)), len(set(continents_filtered)))


    # df_all = pd.read_csv("data/combine_ds_geode_all.csv")
    # df_filtered = df_all.drop(indexes_removed)
    # df_filtered = df_filtered.drop(['Unnamed: 0.3','Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0'], axis=1)
    # df_filtered.to_csv("data/df_all_filtered.csv")
    return indexes


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
    CLIP_feat = [img_CLIP_feat, txt_CLIP_feat]
    SBERT_feat = [caption1_SBERT_feat, caption2_SBERT_feat, caption3_SBERT_feat, caption4_SBERT_feat,
                  caption_all_SBERT_feat]
    return CLIP_feat, SBERT_feat



def balance_country(threshold_object, threshold_country):
    dict_filtered_countries_per_object = {}
    set_countries = set()
    objects, countries, regions, continents, income, images = get_metadata()
    dict_countries_per_object = {}
    for object, country in zip(objects, countries):
        if object not in dict_countries_per_object:
            dict_countries_per_object[object] = []
        dict_countries_per_object[object].append(country)

    for object, countries_object in dict_countries_per_object.items():
        if Counter(countries_object).most_common()[0][1] >= threshold_object:
            dict_filtered_countries_per_object[object] = [country for country in set(countries_object) if
                                                                     countries_object.count(
                                                                         country) >= threshold_country]
            for country in dict_filtered_countries_per_object[object]:
                set_countries.add(country)
            # print(object_to_predict, Counter(countries_object).most_common())
    print(f"#countries: {len(set_countries)}, #objects: {len(dict_filtered_countries_per_object)}")
    print(set_countries)
    print(dict_filtered_countries_per_object.keys())
    # print(dict_filtered_countries_per_object)
    return dict_filtered_countries_per_object


def balance_per_country(max_threshold_object_per_country, min_threshold_object_per_country):
    dict_filtered_countries_per_object = balance_country(max_threshold_object_per_country, min_threshold_object_per_country)
    objects, countries, regions, continents, incomes, images = get_metadata()

    new_data, new_indexes = [], []
    count_object_country = {}
    for index, [object, country, region, continent, income, image] in enumerate(zip(objects, countries, regions, continents, incomes,
                                                                   images)):
        if (object, country) not in count_object_country:
            count_object_country[(object, country)] = 0
        count_object_country[(object, country)] += 1
        if object in dict_filtered_countries_per_object and country in dict_filtered_countries_per_object[object] and \
                count_object_country[(object, country)] <= min_threshold_object_per_country:
            new_data.append([object, country, region, continent, income, image])
            new_indexes.append(index)

    # new_data = sorted(new_data, key=lambda x: (x[0], x[1]))
    # print(new_indexes)
    # df = pd.DataFrame(new_data, columns=['topic', 'country', 'region', 'continent', 'quartile', 'image'])
    # df.to_csv("data/combine_ds_geode_balanced_country.csv")
    return new_indexes

def balance_metadata():
    new_indexes = balance_per_country(max_threshold_object_per_country=100, min_threshold_object_per_country=100)
    #TODO: Balance per region and continent

if __name__ == '__main__':
    balance_metadata()

    # objects, countries, regions, continents, income, images = get_metadata()
    # list_obj_US = []
    # for [object, country] in zip(objects, countries):
    #     if country == "United States":
    #         list_obj_US.append(object)
    # print(Counter(list_obj_US).most_common())
    # filter_data(objects, countries, regions, continents, income, images)
    # get_features()

    # df_all = pd.read_csv("data/combine_ds_geode_all.csv")
    # df_all["quartile"] = df_all["quartile"].fillna("none")
    # df_all.to_csv("data/combine_ds_geode_all.csv")

