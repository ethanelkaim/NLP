from gensim import downloader
import pickle
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


def preprocess(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    words = [sen.split()[0].lower() for sen in sentences if sen.strip()]  # Select first word, make lowercase
    tags = [sen.split()[1] for sen in sentences if len(sen.split()) > 1 and sen.strip()]  # Select the word tag

    return words, tags


def model1(train_path, dev_path, test_path):
    GLOVE_PATH = 'glove-twitter-200'

    try:
        glove = KeyedVectors.load('glove_model.pkl')
    except:
        glove = downloader.load(GLOVE_PATH)
        with open('glove_model.pkl', 'wb') as f:
            pickle.dump(glove, f)

    train_words, train_tags = preprocess(train_path)
    dev_words, dev_tags = preprocess(dev_path)

    X_train, Y_train = [], []
    for word, tag in zip(train_words, train_tags):
        if word in glove.key_to_index:
            X_train.append(glove[word])
            Y_train.append(tag)

    X_dev, Y_dev = [], []
    for word, tag in zip(dev_words, dev_tags):
        if word in glove.key_to_index:
            X_dev.append(glove[word])
            Y_dev.append(tag)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_dev = np.array(X_dev)
    Y_dev = np.array(Y_dev)

    Y_train_binary = [0 if tag == 'O' else 1 for tag in Y_train]
    Y_dev_binary = [0 if tag == 'O' else 1 for tag in Y_dev]

    # model = LogisticRegression(max_iter=200)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train_binary)
    Y_pred = model.predict(X_dev)

    f1 = f1_score(Y_dev_binary, Y_pred)
    print(f"Validation F1 Score: {f1}")



def model3(train_path, dev_path, test_path):
    pass


def main():
    train_path = 'data/train.tagged'
    dev_path = 'data/dev.tagged'
    test_path = 'data/test.tagged'

    model1(train_path, dev_path, test_path)
    model3(train_path, dev_path, test_path)


if __name__ == '__main__':
    main()
