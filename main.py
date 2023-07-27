from collections import Counter

import pandas as pd
path_data_geode = "/nfs/turbo/coe-mihalcea/shared_data/GeoDE"
path_images_geode = "/".join([path_data_geode, "images"])
index_geode = pd.read_csv("/".join([path_data_geode, "index.csv"]))

path_data_ds = "/nfs/turbo/coe-mihalcea/shared_data/dollarstreet"
index_ds = pd.read_csv("/".join([path_data_ds, "results_CLIP.csv"]))

def get_metadata():
    # threshold = 1000
    objects = list(index_geode['object'])
    countries = list(index_geode['ip_country'])
    regions = list(index_geode['region'])
    images = list(index_geode['file_path'])



    # remove data with few datapoints
    # countries_to_remove = [country[0] for country in Counter(countries).items() if country[1] < threshold]
    # indexes = [i for i, x in enumerate(countries) if x not in countries_to_remove]
    indexes = [i for i, x in enumerate(countries)]
    objects_filtered = [objects[i] for i in indexes]
    countries_filtered = [countries[i] for i in indexes]
    regions_filtered = [regions[i] for i in indexes]
    # images_filtered = [images[i] for i in indexes]

    return objects_filtered, countries_filtered, regions_filtered
    # return objects_filtered, countries_filtered, regions_filtered, images_filtered




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_metadata()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
