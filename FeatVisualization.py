import torch
import numpy as np
import h5py
import utils
from src import deal_with_folder, instance_predict
from Dataset import DogDataset

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""


def save_true_features(data, file_name: str, data_name: str, model: torch.nn.Module, feat_dim) -> None:
    """
    Saving the true features into .h5 file so that we can make use of it to visualize.
    """
    from torch.utils.data import DataLoader

    data_len, labels, batch_size = len(data), data.labels.copy(), 1
    labels = np.unique(labels)
    np.random.seed(42)           # seed keeps every model sampling the same random label
    np.random.shuffle(labels)
    sampling2visual = labels[:20]   # get 20 that many labels to visualize randomly

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    hdf5_file = h5py.File(file_name, 'a')
    if data_name not in hdf5_file:
        hdf5_file.create_dataset(data_name, (0, feat_dim), maxshape=(None, feat_dim), dtype=np.float32)
        hdf5_file.create_dataset(f"{data_name}_label", (1,), maxshape=(None,), dtype=int)
    the_storing_features = hdf5_file[data_name]
    the_storing_features_label = hdf5_file[f"{data_name}_label"]

    count = 0
    for i, data in enumerate(train_loader):
        x, y = data
        if (the_y := y.numpy()) in sampling2visual:
            count = count + 1
            print(f"进度: {i + 1}/{data_len}")
            # h5 file accepts numpy array and dons accept tensor
            output, _ = model(x)

            # h5 file must defile the storage, we have to adjust it dynamically.
            the_storing_features.resize((count, feat_dim))
            the_storing_features_label.resize(count, axis=0)

            the_storing_features[count - 1] = output.detach().numpy()
            the_storing_features_label[count - 1] = the_y

    hdf5_file.close()


def read_file(file_name: str, data_name: str) -> (np.array, np.array):
    """
    Parsing .h5 file's feature and corresponding labels.
    """
    with h5py.File(file_name, 'r') as file:
        data, labels = file.get(data_name), file.get(f"{data_name}_label")
        if data:
            return data[:], labels[:]
        else:
            raise Exception("the data name isn't in the file.")


def t_sne(data_x: np.array, data_y: np.array, model_name: str) -> None:
    """
    Using t-SNE algorithm to reduction features into embeddings and saving them after that.
    """
    from openTSNE import TSNE
    from sklearn.model_selection import train_test_split

    x_train, x_test, train_label, test_label = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
    tsne = TSNE(
        perplexity=30, metric="euclidean",
        n_jobs=8, random_state=42, verbose=True,
    )
    emb_train = tsne.fit(x_train)
    emb_test = emb_train.transform(x_test)
    np.save(root + f"{model_name}_embedding_train.npy", emb_train)
    np.save(root + f"{model_name}_embedding_test.npy", emb_test)
    np.save(root + f"{model_name}_y_train.npy", train_label)
    np.save(root + f"{model_name}_y_test.npy", test_label)


def draw_figure(data, emb_train: np.array, emb_test: np.array,
                train_label: np.array, test_label: np.array,
                model_name: str, save_fig: bool = False) -> None:
    """
    Plotting these embeddings to observe whether features has clustered.
    """
    import matplotlib.pyplot as plt

    uniq_text_labels = data.classes_name[(u_digits_label := np.unique(train_label))]

    # mapping digits label to text label (unique)
    label_mapping = dict(zip(u_digits_label, uniq_text_labels))

    # do the mapping for non-unique digital label
    str_train_labels = [label_mapping[digit_label] for digit_label in train_label]
    str_test_labels = [label_mapping[digit_label] for digit_label in test_label]

    color_mapping = dict(zip(uniq_text_labels, utils.ZEISEL_COLORS))

    fig, ax = plt.subplots(figsize=(25, 20))
    utils.plot(emb_train, str_train_labels, colors=color_mapping, alpha=1, ax=ax)
    utils.plot(emb_test, str_test_labels, colors=color_mapping,  alpha=0.5, ax=ax)
    plt.show()
    if save_fig:
        fig.savefig(f'./visualize_file/{model_name}_feature_visualize.png',
                    dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    root = deal_with_folder('./visualize_file/')
    the_model, dim = 'GoogLeNetForDogDetection', 1024
    m = torch.load(f'saved_model/{the_model}_model.pth')
    data_set = DogDataset(use_type='test')
    m.eval()

    visualize_prediction = 1
    visualize_feature = 1 - visualize_prediction

    if visualize_feature:
        save_true_features(data_set, root + f'{the_model}Features.h5', 'features', m, dim)

        X, Y = read_file(root + f'{the_model}Features.h5', 'features')
        print("the saved low dimensions feature: ", X.shape)
        t_sne(X, Y, the_model)

        embedding_train = np.load(root + f"{the_model}_embedding_train.npy")
        embedding_test = np.load(root + f"{the_model}_embedding_test.npy")
        y_train, y_test = np.load(root + f"{the_model}_y_train.npy"), np.load(root + f"{the_model}_y_test.npy")

        draw_figure(data_set, embedding_train, embedding_test, y_train, y_test, the_model)

    if visualize_prediction:
        # test box idx: 111, 333, 222, 444, 555
        img, box, label = data_set.get_dataset_box(3324)
        # detection task
        instance_predict(model=m, data_set=data_set, image_file=img,
                         detection=True, true_box=box, true_cls=label)
        # classification task
        # instance_predict(model=m, data_set=data_set, image_file=img)


author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""