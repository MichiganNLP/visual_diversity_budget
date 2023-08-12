from collections import Counter
from random import randint

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from main import get_metadata, get_features, filter_data, balance_per_country
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


def SVM(feat_test, feat_train, labels_train):
    sc = StandardScaler()
    feat_test = sc.fit_transform(feat_test)
    feat_train = sc.fit_transform(feat_train)
    # print(f"All feat_train.shape: {feat_train.shape}")

    method = svm.SVC(class_weight="balanced")
    method.fit(feat_train, labels_train)
    predicted = method.predict(feat_test)
    return predicted


# targets: country, region, continent or income
def predict_object(model_name, object_to_predict, targets, features):
    list_results_per_object = []
    remove_object = False
    print(f"Running {model_name} ..")
    for target_to_predict in sorted(set(targets)):
        labels = [1 if target == target_to_predict else 0 for target in targets]

        feat_train, feat_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1,
                                                                            random_state=7)

        if all(int(num) == 0 for num in labels_test):
            min_class_f1 = 0
            random_class_f1 = 0
        else:
            # maj_class_f1 = f1_score(labels_test, [0] * len(labels_test)) * 100
            # min_class_f1 = f1_score(labels_test, [1] * len(labels_test)) * 100
            random_labels = [randint(0, 1) for _ in range(len(labels_test))]
            random_class_f1 = f1_score(labels_test, random_labels) * 100
        if len(set(labels_train)) < 2:
            print(f"Only one class for {object_to_predict} in country/ region/ continent/ income {target_to_predict}!!")
            print(f"Removing {object_to_predict} ..")
            remove_object = True
            continue

        if model_name == "SVM":
            predicted = SVM(feat_test, feat_train, labels_train)
        else:
            raise ValueError(f"no known model {model_name}")

        if all(int(num) == 0 for num in predicted):
            method_f1 = 0
        else:
            method_f1 = round(f1_score(labels_test, predicted) * 100, 2)

        # list_results_per_object.append([target_to_predict, round(min_class_f1, 1), round(method_f1, 1)])
        list_results_per_object.append([target_to_predict, round(random_class_f1, 1), round(method_f1, 1)])

    return list_results_per_object, remove_object



def feature_ablation(CLIP_feat, SBERT_feat, indexes_object):
    '''
        CLIP features
    '''
    [img_CLIP_feat, txt_CLIP_feat] = CLIP_feat
    img_CLIP_feat_filtered = np.take(img_CLIP_feat, indexes_object, axis=0)
    txt_CLIP_feat_filtered = np.take(txt_CLIP_feat, indexes_object, axis=0)
    '''
        Caption features
    '''
    [caption1_SBERT_feat, caption2_SBERT_feat, caption3_SBERT_feat, caption4_SBERT_feat,
     caption_all_SBERT_feat] = SBERT_feat
    caption_all_SBERT_feat_filtered = np.take(caption_all_SBERT_feat, indexes_object, axis=0)
    caption1_SBERT_feat_filtered = np.take(caption1_SBERT_feat, indexes_object, axis=0)
    caption2_SBERT_feat_filtered = np.take(caption2_SBERT_feat, indexes_object, axis=0)
    caption3_SBERT_feat_filtered = np.take(caption3_SBERT_feat, indexes_object, axis=0)
    caption4_SBERT_feat_filtered = np.take(caption4_SBERT_feat, indexes_object, axis=0)

    '''
         Feature Ablations
    '''
    # features = np.random.rand(len(indexes), 512) #TODO: Random Features Baseline?

    # features = img_CLIP_feat_filtered
    # features = caption_SBERT_feat_filtered
    features = np.concatenate((caption_all_SBERT_feat_filtered, img_CLIP_feat_filtered), axis=1)
    # print(caption_SBERT_feat_filtered.shape, img_CLIP_feat_filtered.shape, features.shape)

    return features


def run_diversity_classifier(model_name):
    objects, countries, regions, continents, income, images = get_metadata()
    indexes_filtered = balance_per_country(max_threshold_object_per_country=100, min_threshold_object_per_country=100)
    # indexes_filtered = balance_per_region()
    # indexes_filtered = balance_per_continent()
    CLIP_feat, SBERT_feat = get_features()

    all_results = {}
    objects_filtered = [objects[i] for i in indexes_filtered]
    countries_filtered = [countries[i] for i in indexes_filtered]

    set_objects = sorted(set(objects_filtered)) - set(['visit', 'dog'])
    set_countries = sorted(set(countries_filtered))

    # # object_to_predict = "bag"
    # For each object, predict the region
    for object_to_predict in tqdm(list(set_objects)):
        # Filter countries, region, continents, income targets by object to predict
        indexes_object = [i for i, x in enumerate(objects) if x == object_to_predict and i in indexes_filtered]
        countries_object = [countries[i] for i in indexes_object]

        # print(f"Object: {object_to_predict}, counter: {Counter(countries_object).most_common()}")
        # regions_object = [regions[i] for i in indexes_object]
        # continents_object = [continents[i] for i in indexes_object]
        # income_object = [income[i] for i in indexes_object] #TODO - only DS for now, GeoDE = none

        features_object = feature_ablation(CLIP_feat, SBERT_feat, indexes_object)

        '''
            1 vs. all binary classification
            TODO: Change targets: countries_object/ regions_object/ continents_object/ income_object
        '''
        results_per_object, remove_object = predict_object(model_name=model_name, object_to_predict=object_to_predict,
                                            targets=countries_object, features=features_object)

        if remove_object:
            continue
        '''
            Analyse results
        '''
        print(f"Show visual diversity for topic {object_to_predict}:")
        for [target, baseline_f1, method_f1] in results_per_object:
            if object_to_predict not in all_results:
                all_results[object_to_predict] = []
            all_results[object_to_predict].append([target, method_f1, baseline_f1, round(method_f1-baseline_f1, 1)])
            # if baseline_f1 < method_f1:
            #     print(f"target: {target}, baseline_f1: {baseline_f1}, method_f1: {method_f1}")

        countries_no_objects = set(set_countries) - set(countries_object) # Countries not present for that object
        for c in countries_no_objects:
            all_results[object_to_predict].append([c, 'nan', 'nan', 'nan'])

    all_results_df = []
    for object, list_results in all_results.items():
        sorted_results = sorted(list_results, key=lambda x: x[0])  # sort countries alphabetically
        for [country, method_f1, baseline_f1, diff] in sorted_results:
            all_results_df.append([object, country, method_f1, baseline_f1, diff])


    df = pd.DataFrame(all_results_df, columns=['Topic', 'Country', 'SVM_F1', 'Random_F1', 'Diff'])
    df.to_csv("data/results_countries.csv")

    # df = pd.DataFrame(all_results, columns=['Topic', 'Region', 'SVM_F1', 'Random_F1', 'Diff'])
    # df.to_csv("data/results_regions.csv")

if __name__ == '__main__':
    run_diversity_classifier(model_name="SVM")
    # analyse_results(model_name="SVM")
