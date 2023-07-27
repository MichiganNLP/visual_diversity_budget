from collections import Counter

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from main import get_metadata, get_features
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
def predict_object(method_name, object_to_predict, targets, features):
    list_results_per_object = []
    print(f"Running {method_name} ..")
    for target_to_predict in sorted(set(targets)):
        labels = [1 if target == target_to_predict else 0 for target in targets]

        feat_train, feat_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=7)

        if all(int(num) == 0 for num in labels_test):
            min_class_f1 = 0
        else:
            min_class_f1 = f1_score(labels_test, [1] * len(labels_test)) * 100
        if len(set(labels_train)) < 2:
            print(f"Skipping object {object_to_predict} in country/ region/ continent/ income {target_to_predict}")
            continue

        if method_name == "SVM":
            predicted = SVM(feat_test, feat_train, labels_train)
        else:
            raise ValueError(f"no known method {method_name}")

        if all(int(num) == 0 for num in predicted):
            method_f1 = 0
        else:
            method_f1 = round(f1_score(labels_test, predicted) * 100, 2)

        list_results_per_object.append([target_to_predict, round(min_class_f1, 2), round(method_f1, 2)])

    return list_results_per_object



def classifier(method_name): #TODO: test poly kernel, fine-tune C, gamma ..
    img_CLIP_feat, txt_CLIP_feat, caption_SBERT_feat = get_features()
    objects, countries, regions = get_metadata()

    set_objects = sorted(set(objects))
    # For each object, predict the region
    # for object_to_predict in tqdm(set_objects):
    object_to_predict = "bag"
    indexes = [i for i, x in enumerate(objects) if x == object_to_predict]
    objects_filtered = [objects[i] for i in indexes]
    countries_filtered = [countries[i] for i in indexes]
    regions_filtered = [regions[i] for i in indexes]

    '''
    CLIP features
    '''
    img_CLIP_feat_filtered = np.take(img_CLIP_feat, indexes, axis=0)
    # txt_CLIP_feat_filtered = np.take(txt_CLIP_feat, indexes, axis=0)
    '''
    Caption features
    '''
    caption_SBERT_feat_filtered = np.take(caption_SBERT_feat, indexes, axis=0)

    '''
         Feature Ablations
    '''
    # features = img_CLIP_feat_filtered
    # features = caption_SBERT_feat_filtered
    # features = np.random.rand(len(indexes), 512) #TODO: Random Features
    features = np.concatenate((caption_SBERT_feat_filtered, img_CLIP_feat_filtered), axis=1)
    # print(caption_SBERT_feat_filtered.shape, img_CLIP_feat_filtered.shape, features.shape)

    '''
        1 vs. all binary classification
    '''
    results_per_object = predict_object(method_name=method_name, object_to_predict=object_to_predict, targets=regions_filtered, features=features)

    '''
        Analyse results
    '''
    print(f"Show diversity for topic {object_to_predict} :")
    for [target, baseline_f1, method_f1] in results_per_object:
        if baseline_f1 < method_f1:
            print(f"target: {target}, baseline_f1: {baseline_f1}, method_f1: {method_f1}")


if __name__ == '__main__':
    classifier(method_name="SVM")
    # analyse_results(method_name="SVM")