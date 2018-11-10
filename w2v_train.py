from tensorflow.keras import layers, Model, optimizers
import pandas as pd
import os
import numpy as np


def build_keras_model(cat_dims: list, attributes_dims: list, embedding_length=32):
    """

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

    attributes_embeddings = [layers.Embedding(input_length=1, input_dim=attr_dim, output_dim=embedding_length) for
                             attr_dim in
                             attributes_dims]

    embedded_cat = [layers.Activation("sigmoid")(cat_embeddings[i](cat_inputs[i])) for i in
                    range(len(cat_dims))]

    embedded_cat_next = [layers.Activation("sigmoid")(cat_embeddings[i](cat_next_inputs[i])) for i in
                         range(len(cat_dims))]

    embedded_attr = [layers.Activation("sigmoid")(attributes_embeddings[i](attributes_inputs[i])) for i in
                     range(len(attributes_dims))]

    embedded_all = embedded_cat + embedded_cat_next + embedded_attr

    flattened_cat = [layers.Flatten(name="flattened_{}".format(i))(embedded_all[i]) for i in range(len(embedded_all))]

    concatenate_layer = layers.Concatenate(axis=-1)(flattened_cat)
    output = layers.Dense(1, activation="sigmoid", name="output")(concatenate_layer)
    model = Model(inputs=cat_inputs + cat_next_inputs + attributes_inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(), loss="mean_squared_error")
    model.summary()
    return model


def load_data() -> [pd.DataFrame, pd.DataFrame]:
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
    info_data = data[1].sort_values(by="user_id").values
    while True:
        if (index + windows_size + 1) > len(data[0]):  # start from head when reaching the end
            index = windows_size
        batch = []
        label = np.zeros(shape=(batch_size), dtype=np.int8)
        batch_num = 0
        true_num = 0
        while batch_num < batch_size:
            if true_num < true_needed:
                batch_next = np.concatenate((b_data[index - windows_size:index],
                                             b_data[index + 1:index + 1 + windows_size]))
                centre_line = b_data[index]
                label_part = (batch_next[:, 0] == centre_line[0]).astype(int)
                label[batch_num:batch_num + windows_size * 2] = label[
                                                                batch_num:batch_num + windows_size * 2] + label_part
                full_length_batch_first = np.repeat(centre_line[1:].reshape((1, 1, -1)), len(batch_next), axis=0)
                cat_connected = np.concatenate(
                    (full_length_batch_first, batch_next[:, 1:].reshape(len(batch_next), 1, -1)), axis=1)
                attrs = np.repeat(info_data[centre_line[0]].reshape((1, 1, -1)), len(batch_next), axis=1)
                # No idea how to handle this : full = np.concatenate((cat_connected,attrs), axis=1)
                full = [i + [attrs[0][0].tolist()] for i in cat_connected.tolist()]
                # up
                batch = batch + full
                batch_num += 2 * windows_size
                index += skip_num
            else:
                # negative examples is possible to be positive
                start_index = np.random.randint(len(data[0]))
                next_index = np.random.randint(len(data[0]))
                while abs(next_index - start_index) < batch_size - batch_num:
                    next_index = np.random.randint(len(data[0]))

                # this can speed up
                for i in range(batch_size - batch_num):
                    batch.append(np.concatenate((
                        b_data[start_index + i][1:].reshape(-1, 1),
                        b_data[next_index + i][1:].values.reshape(-1, 1))).tolist()
                                 +
                                 [info_data[b_data[start_index + i][0]].tolist()])
                    # label.append(0)
                batch_num += 1
        yield batch, label


if __name__ == '__main__':
    # build_keras_model([6400, 6400, 6400], [40, 50])
    # print(load_data()[0].iloc[10 + 4,][["cat_id", "seller_id", "brand_id"]].values.reshape(-1, 1))
    # print(load_data()[1].loc[189057])

    import time

    data = load_data()
    k = 0
    start = time.time()
    for i in batch_generator(data):
        k += 1
        print(i)
        if k == 1000:
            break
    finish = time.time()
    print((finish - start))

    # pass
