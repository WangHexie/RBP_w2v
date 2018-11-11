from tensorflow.keras import layers, Model, optimizers
from tensorflow.train import AdamOptimizer
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import urllib.request


def build_keras_model(cat_dims: list, attributes_dims: list, embedding_length=32, attr_embedding_length=8):
    """

    :param attr_embedding_length:
    :param cat_dims: list of int:[x1, x2, x3]
    :param attributes_dims: list of dims(int):[y1, y2, y3]
    :param embedding_length:
    :return:
    """
    if cat_dims is None:
        cat_dims = []
    cat_inputs = [layers.Input(shape=(1,), name="cat_input_{}".format(i)) for i in range(len(cat_dims))]
    cat_next_inputs = [layers.Input(shape=(1,), name="cat_next_input_{}".format(i)) for i in range(len(cat_dims))]
    attributes_inputs = [layers.Input(shape=(1,), name="attributes_input{}".format(i)) for i in
                         range(len(attributes_dims))]

    cat_embeddings = [layers.Embedding(input_length=1, input_dim=cat_dim, output_dim=embedding_length) for cat_dim in
                      cat_dims]

    attributes_embeddings = [layers.Embedding(input_length=1, input_dim=attr_dim, output_dim=attr_embedding_length) for
                             attr_dim in
                             attributes_dims]

    embedded_cat = [layers.Activation("relu")(cat_embeddings[i](cat_inputs[i])) for i in
                    range(len(cat_dims))]

    embedded_cat_next = [layers.Activation("relu")(cat_embeddings[i](cat_next_inputs[i])) for i in
                         range(len(cat_dims))]

    embedded_attr = [layers.Activation("relu")(attributes_embeddings[i](attributes_inputs[i])) for i in
                     range(len(attributes_dims))]

    embedded_all = embedded_cat + embedded_cat_next + embedded_attr

    flattened_cat = [layers.Flatten(name="flattened_{}".format(i))(embedded_all[i]) for i in range(len(embedded_all))]

    concatenate_layer = layers.Concatenate(axis=-1)(flattened_cat)
    # dense_layer = layers.Dense(70, activation="sigmoid")(concatenate_layer)
    output = layers.Dense(1, activation="relu", name="output")(concatenate_layer)
    model = Model(inputs=cat_inputs + cat_next_inputs + attributes_inputs, outputs=output)
    model.compile(optimizer=AdamOptimizer(), loss="mean_squared_error")
    model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(
                tpu='grpc://' + '10.0.101.2'
        )
    )
    model.summary()
    return model


def load_data() -> [pd.DataFrame, pd.DataFrame]:
    exists = os.path.isfile(os.path.join("data", "data.pkl")) and os.path.isfile(os.path.join("data", "user_info.pkl"))
    if not exists:
        print("files don't exist, downloading now..............")
        dict_url = "https://github.com/WangHexie/RBP_w2v/releases/download/v0.1/id_dict.txt"
        data_url = "https://github.com/WangHexie/RBP_w2v/releases/download/v0.1/data.pkl"
        info_url = "https://github.com/WangHexie/RBP_w2v/releases/download/v0.1/user_info.pkl"
        urllib.request.urlretrieve(dict_url, os.path.join("data", "id_dict.txt"))
        urllib.request.urlretrieve(data_url, os.path.join("data", "data.pkl"))
        urllib.request.urlretrieve(info_url, os.path.join("data", "user_info.pkl"))

    return [pd.read_pickle(os.path.join("data", "data.pkl"), compression='zip').fillna(0),
            pd.read_pickle(os.path.join("data", "user_info.pkl"), compression='zip')]

    # return [pd.read_pickle(os.path.join("data", "data.pkl"), compression='zip').fillna(0),
    #         pd.get_dummies(pd.read_pickle(os.path.join("data", "user_info.pkl"), compression='zip'),
    #                        columns=["age_range", "gender"])]


def batch_generator(data: [pd.DataFrame, pd.DataFrame], batch_size=64, negative_percent=0.5, windows_size=1,
                    skip_num=2):
    """

    :param skip_num:
    :param data: [train_data, attributes_data]
    :param batch_size:
    :param negative_percent:
    :param windows_size:
    :return:
    """
    index = windows_size
    true_needed = int(batch_size * negative_percent)
    b_data = data[0][["user_id", "seller_id", "cat_id", "brand_id"]].values
    info_data = data[1].values
    while True:
        batch = np.empty(shape=(0, 8, 1), dtype=np.int32)
        label = np.zeros(shape=batch_size, dtype=np.int8)
        batch_num = 0
        true_num = 0
        while batch_num < batch_size:
            if (index + windows_size + 1) > len(data[0]):  # start from head when reaching the end
                index = windows_size

            if true_num < true_needed:
                batch_next = np.concatenate((b_data[index - windows_size:index],
                                             b_data[index + 1:index + 1 + windows_size]))
                centre_line = b_data[index]
                label_part = (batch_next[:, 0] == centre_line[0]).astype(int)
                label[batch_num:batch_num + windows_size * 2] = label[
                                                                batch_num:batch_num + windows_size * 2] + label_part
                full_length_batch_first = np.repeat(centre_line[1:].reshape((1, -1, 1)), len(batch_next), axis=0)
                cat_connected = np.concatenate(
                    (full_length_batch_first, batch_next[:, 1:].reshape(len(batch_next), -1, 1)), axis=1)
                attrs = np.repeat(info_data[centre_line[0]].reshape((1, -1, 1)), len(batch_next), axis=0)
                full = np.concatenate((cat_connected, attrs), axis=1)

                batch = np.concatenate((batch, full), axis=0)
                batch_num += 2 * windows_size
                index += skip_num
            else:
                # negative examples is possible to be positive
                start_index = np.random.randint(len(data[0]))
                next_index = np.random.randint(len(data[0]))
                while abs(next_index - start_index) < batch_size - batch_num:
                    next_index = np.random.randint(len(data[0]))

                cat_connected = np.concatenate((
                    b_data[start_index:start_index + batch_size - batch_num][1:].reshape(1, -1, 1),
                    b_data[next_index:next_index + batch_size - batch_num][1:].reshape(1, -1, 1)), axis=1)
                attrs = info_data[b_data[start_index:start_index + start_index + batch_size - batch_num][0]].reshape(
                    (1, -1, 1))
                full = np.concatenate((cat_connected, attrs), axis=1)
                batch = np.concatenate((batch, full), axis=0)
                batch_num += 1
        batch = batch.reshape((batch_size, 8)).transpose()
        x = (dict(
            [('cat_input_{}'.format(i), batch[i]) for i in range(3)] + [('cat_next_input_{}'.format(i), batch[i + 3])
                                                                        for i in range(3)] +
            [('attributes_input{}'.format(i), batch[i + 6]) for i in range(2)]
        ), dict([('output', label)]))
        yield x


def train():
    batch_size = 64
    data = load_data()
    cat_dims = [len(set(data[0][name])) for name in ["seller_id", "cat_id", "brand_id"]]
    attr_dims = [len(set(data[1][name])) for name in ["age_range", "gender"]]
    model = build_keras_model(cat_dims, attr_dims, embedding_length=64)
    model.fit_generator(batch_generator(data=data, batch_size=batch_size),
                        steps_per_epoch=int(len(data[0]) / batch_size), epochs=10)


if __name__ == '__main__':
    train()
