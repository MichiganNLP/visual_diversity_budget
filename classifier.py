from collections import Counter

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from main import path_data_geode, index_geode, get_metadata
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


def get_features():
    img_CLIP_feat = np.load(path_data_geode + "/image_CLIP.npy")
    txt_CLIP_feat = np.load(path_data_geode + "/txt_CLIP.npy")
    # similarity_CLIP = np.load(path_data_geode + "/similarity_CLIP.npy")
    # similarity_CLIP = np.load(path_data_geode + "/similarity_CLIP.npy")
    caption1_SBERT_feat = np.load(path_data_geode + "/caption1_SBERT.npy")
    # caption2_SBERT_feat = np.load(path_data_geode + "/caption2_SBERT.npy")

    return img_CLIP_feat, txt_CLIP_feat, caption1_SBERT_feat

# targets: country, region, continent or income
def predict_object(method_name, object_to_predict, targets, features):
    list_results_per_object = []
    for target_to_predict in sorted(set(targets)):
        labels = [1 if target == target_to_predict else 0 for target in targets]

        feat_train, feat_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=7)
        # maj_class_acc = accuracy_score(labels_test, [0] * len(labels_test)) * 100
        # maj_class_f1 = f1_score(labels_test, [0] * len(labels_test)) * 100
        # min_class_acc = accuracy_score(labels_test, [1] * len(labels_test)) * 100
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
        # method_accuracy = round(accuracy_score(labels_test, predicted) * 100, 2)
        if all(int(num) == 0 for num in predicted):
            method_f1 = 0
        else:
            method_f1 = round(f1_score(labels_test, predicted) * 100, 2)
        # print(f"Object: {object_to_predict}, Country: {target_to_predict}, Method {method_name}, Acc: {method_accuracy:.1f}, F1: {method_f1:.1f}")
        # print(f"Maj class baseline Acc: {maj_class_acc:.1f}, F1: {maj_class_f1:.1f}")
        # print(f"Min class baseline Acc: {min_class_acc:.1f}, F1: {min_class_f1:.1f}")
        # print("---------------------------------------")

        list_results_per_object.append([target_to_predict, round(min_class_f1, 2), round(method_f1, 2)])
    return list_results_per_object



def classifier(method_name): #TODO: test poly kernel, fine-tune C, gamma ..
    results = []
    specific_objects_country,  specific_objects_region, specific_objects_country = [], [], []
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
    print(caption_SBERT_feat_filtered.shape, img_CLIP_feat_filtered.shape, features.shape)

    '''
        1 vs. all binary classification
    '''
    results_per_object = predict_object(method_name=method_name, object_to_predict=object_to_predict, targets=countries_filtered, features=features)

    '''
        Analyse results
    '''
    for [target, baseline_f1, method_f1] in results_per_object:
        if baseline_f1 < method_f1:
            print(f"target: {target}, baseline_f1: {baseline_f1}, method_f1: {method_f1}")


if __name__ == '__main__':
    classifier(method_name="SVM")
    # analyse_results(method_name="SVM")