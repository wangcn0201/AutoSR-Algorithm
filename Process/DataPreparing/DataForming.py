import numpy as np
import torch
import pandas as pd

class Former:
    def __init__(self, user_item_sequence, num_users, num_items):
        user_ids, item_ids = [], []
        for uid, item_seq in enumerate(user_item_sequence):
            for iid in item_seq:
                user_ids.append(uid)
                item_ids.append(iid)

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        self.num_users = num_users
        self.num_items = num_items

        self.user_ids = user_ids
        self.item_ids = item_ids

        self.sequences = None
        self.test_sequences = None


    def to_sequence(self, sequence_length=5, target_length=1):    #返回序列
        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        self.num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])
        num_subsequences = self.num_subsequences

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid

        self.sequences = sequences
        self.test_sequences = test_sequences
        self.sequences_users = sequence_users
        return sequences, test_sequences, sequences_targets, sequence_users

    def get_history_item_users(self, data_path):
        col_names = ['user_id', 'item_id', 'timestamp']
        data_records = pd.read_csv(data_path, sep='\t', names=col_names, engine='python')

        self.gotten_items = set()
        self.item_to_pastusers = {}
        self.time_for_users_to_item = {}

        for i in range(len(data_records)):
            user_id = data_records["user_id"][i]
            item_id = data_records["item_id"][i]

            if item_id not in self.gotten_items:
                self.gotten_items.add(item_id)
                self.item_to_pastusers[item_id] = [user_id]
                self.time_for_users_to_item[item_id] = {user_id: 0}
            else:
                self.item_to_pastusers[item_id].append(user_id)
                self.time_for_users_to_item[item_id][user_id] = len(self.item_to_pastusers[item_id]) - 1


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq)

def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')