import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import csv
import gzip


def make_datafiles(path = "../Data/Amazon/", data_name = "video", mincount=10):
    path = path + data_name + "/"
    data_path = data_name + ".csv"
    if(data_path == "ml100k.csv"):
        data = Ml100k(path, data_path)
    else:
        data = Amazon(path, data_path, mincount)
    df_ordered, user_inverse_mapping, item_inverse_mapping, item_mapping = data.init_data(write=True)
    data.generate_extra_info(path, df_ordered, user_inverse_mapping, item_inverse_mapping)
    data.generate_meta_info(data_name, item_inverse_mapping, item_mapping)
    print("data_name:" , data_name)
    print("users number:", len(user_inverse_mapping))
    print("items number:", len(item_inverse_mapping))
    print("interactions:", len(df_ordered))

class Amazon:
    def __init__(self, path, data, mincount=10):
        self.mincount = mincount
        self.path = path
        self.data_name = data[0:(len(data)-4)]
        col_names = ['item_id', 'user_id', 'rating', 'timestamp']
        if self.data_name == "ML100K":
            self.data_records = pd.read_csv(path + data, sep='\t', names=col_names, engine='python')
        else:
            self.data_records = pd.read_csv(path + data, sep=',', names=col_names, engine='python')
        # print(len(self.data_records['user_id'].value_counts()), len(self.data_records['item_id'].value_counts()))


    def init_data(self, write=True):
        data_records = self.data_records
        data_records.loc[data_records.rating < 4, 'rating'] = 0
        data_records.loc[data_records.rating >= 4, 'rating'] = 1
        data_records = data_records[data_records.rating > 0]

        min_counts = 5
        df1 = deepcopy(data_records)
        # counts = df1['item_id'].value_counts()
        # df1 = df1[df1["item_id"].isin(counts[counts >= min_counts].index)]
        # counts = df1['user_id'].value_counts()
        # df1 = df1[df1["user_id"].isin(counts[counts >= min_counts].index)]
        #
        # item_list = df1['item_id'].unique()
        # item_set = set(item_list)

        df2 = deepcopy(df1)
        df_ordered = df2.sort_values(['timestamp'], ascending=True)
        order_used_for_get = deepcopy(df_ordered)
        data = df_ordered.groupby('user_id')['item_id'].apply(list)
        unique_data = df_ordered.groupby('user_id')['item_id'].nunique()
        seq_data = data[unique_data[unique_data >= self.mincount].index]
        user_item_dict = seq_data.to_dict()

        new_order = get_order(order_used_for_get, user_item_dict)


        user_mapping = []
        item_set = set()
        for user_id, item_list in seq_data.iteritems():
            user_mapping.append(user_id)
            for item_id in item_list:
                item_set.add(item_id)
        item_mapping = list(item_set)

        user_inverse_mapping = dict()
        for inner_id, true_id in enumerate(user_mapping):
            user_inverse_mapping[true_id] = inner_id

        item_inverse_mapping = dict()
        for inner_id, true_id in enumerate(item_mapping):
            item_inverse_mapping[true_id] = inner_id

        inner_user_records = []
        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_item_dict[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        if write == True:
            save_obj(inner_user_records, self.path + self.data_name + '_item_sequences')
            save_obj(user_mapping, self.path + self.data_name + '_user_mapping')
            save_obj(item_mapping, self.path + self.data_name + '_item_mapping')

        df_ordered.reset_index(drop=True, inplace=True)
        return new_order, user_inverse_mapping, item_inverse_mapping, item_mapping
        # return seq_data, user_mapping, item_mapping, user_inverse_mapping, item_inverse_mapping


    def generate_meta_info(self, data_name, item_inverse_mapping, item_mapping):
        meta_path = "../Data/Amazon/" + data_name + "/" + "meta_" + data_name + ".json.gz"
        meta_df = get_df(meta_path)
        # print(meta_df.head()
        useful_meta_df = meta_df[meta_df['asin'].isin(item_mapping)]
        useful_meta_df = useful_meta_df.reset_index(drop=True)
        # print(useful_meta_df.head())

        l2_cate_lst = list()
        for cate_lst in useful_meta_df['category']:
            l2_cate_lst.append(cate_lst[2] if len(cate_lst) > 2 else np.nan)
        useful_meta_df['l2_category'] = l2_cate_lst
        l2_cates = sorted(useful_meta_df['l2_category'].dropna().unique())
        l2_dict = dict(zip(l2_cates, range(1, len(l2_cates) + 1)))
        useful_meta_df['l2_category'] = useful_meta_df['l2_category'].apply(lambda x: l2_dict[x] if x == x else 0)

        item_meta_data = dict()
        for idx in range(len(useful_meta_df)):
            info = useful_meta_df.iloc[idx]
            item_meta_data[idx] = {
                'item_id': item_inverse_mapping[useful_meta_df.iloc[idx]['asin']],
                'category': useful_meta_df.iloc[idx]['l2_category'],
                'r_complement': map_the_list(item_inverse_mapping, info["also_buy"]) if 'also_buy' in info else [],
                'r_substitute': map_the_list(item_inverse_mapping, info["also_view"]) if 'also_view' in info else [],
            }

        item_meta_df = pd.DataFrame.from_dict(item_meta_data, orient='index')
        item_meta_df = item_meta_df[['item_id', 'category', 'r_complement', 'r_substitute']]
        # print(item_meta_df.head())
        path = "../Data/Amazon/" + data_name + "/"
        item_meta_df.to_csv(path + 'item_meta.csv', sep='\t', index=False)

    def generate_extra_info(self, path, new_order, user_inverse_mapping, item_inverse_mapping):
        rows = []
        for i in range(len(new_order)):
            user = new_order[i][0]
            item = new_order[i][1]
            time = new_order[i][2]
            user_id = user_inverse_mapping[user]
            item_id = item_inverse_mapping[item]
            rows.append(tuple([user_id, item_id, time]))
        with open(path + 'train.csv', 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(rows)


class Ml100k(Amazon):
    def __init__(self, path, data):
        super(Ml100k, self).__init__(path, data)
        col_names = ['user_id', 'item_id', 'rating', 'timestamp']
        self.data_records = pd.read_csv(path + data, sep='\t', names=col_names, engine='python')

def map_the_list(item_inverse_mapping, info_list):
    id_list = []
    for true_id in info_list:
        if true_id not in item_inverse_mapping:
            continue
        else:
            id_list.append(item_inverse_mapping[true_id])
    return id_list


def get_order(df_order, seq_data):
    new_order = []
    df_order.reset_index(drop=True, inplace=True)
    for i in range(len(df_order)):
        uid = df_order["user_id"][i]
        if uid in seq_data:
            iid = df_order["item_id"][i]
            time = df_order["timestamp"][i]
            new_order.append([uid, iid, time])
    return new_order



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f, encoding='latin1')



def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')




if __name__ == '__main__':
    path = "../Data/Amazon/"
    data = "Movies_and_TV"
    mincount = 10

    make_datafiles(path, data, mincount)
    """
    min_count
    numUser
    numItem
    Interactions
    """
