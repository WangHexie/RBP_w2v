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
    attributes_inputs = [layers.Input(shape=(attributes_dims[i],), name="attributes_input{}".format(i)) for i in
                         range(len(attributes_dims))]

    embeddings = [layers.Embedding(input_length=1, input_dim=cat_dim, output_dim=embedding_length) for cat_dim in
                  cat_dims]

    embedded_cat = [layers.Activation("sigmoid")(embeddings[i](cat_inputs[i])) for i in
                    range(len(cat_dims))]

    embedded_cat_next = [layers.Activation("sigmoid")(embeddings[i](cat_next_inputs[i])) for i in
                         range(len(cat_dims))]

    embedded_all = embedded_cat + embedded_cat_next

    flattened_cat = [layers.Flatten(name="flattened_{}".format(i))(embedded_all[i]) for i in range(len(embedded_all))]

    concatenate_layer = layers.Concatenate(axis=-1)(flattened_cat + attributes_inputs)
    output = layers.Dense(1, activation="sigmoid", name="output")(concatenate_layer)
    model = Model(inputs=cat_inputs + cat_next_inputs + attributes_inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(), loss="mean_squared_error")
    model.summary()
    return model


def load_data() -> [pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(os.path.join("data", "data.pkl"), compression='zip').fillna(0),
            pd.get_dummies(pd.read_pickle(os.path.join("data", "user_info.pkl"), compression='zip'),
                           columns=["age_range", "gender"])]


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
    while True:
        batch = []
        label = []
        batch_num = 0
        true_num = 0
        while batch_num < batch_size:
            if true_num < true_needed:
                batch_next = data[0].loc[index - windows_size:index - 1, ].append(
                    data[0].loc[index + 1:index + windows_size, ])
                centre_line = data[0].loc[index]
                for data_row in batch_next.itertuples():
                    if data_row.user_id == centre_line["user_id"]:
                        label.append(1)
                        true_num += 1
                    else:
                        label.append(0)
                    batch.append(
                        list(np.concatenate((np.array(data_row[3:6]).reshape((-1, 1)),
                                            np.array(centre_line[["cat_id", "seller_id", "brand_id"]].values).reshape(
                                                (-1, 1))))) + data[1].iloc[
                            centre_line["user_id"], ].values.tolist())
                batch_num += 2 * windows_size
                index += skip_num
            else:
                batch_num += 1
                pass

        yield batch, label


if __name__ == '__main__':
    # build_keras_model([6400, 6400, 6400], [40, 50])
    # print(load_data())
    # print(load_data()[1].loc[1][["cat_id", "seller_id", "brand_id"]])
    for i in batch_generator(load_data()):
        print(i)
        break
