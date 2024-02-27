from gensim import downloader
import pickle
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

from abc import ABC
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange


GLOVE_PATH = 'glove-twitter-200'


def embedding(train_path, dev_path):
    try:
        glove = KeyedVectors.load('glove_model.pkl')
    except FileNotFoundError:
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

    return X_train, Y_train_binary, X_dev, Y_dev_binary


def preprocess(path, condition=False):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    words = [sen.split()[0].lower() for sen in sentences if sen.strip()]  # Select first word, make lowercase
    tags = [sen.split()[1] for sen in sentences if len(sen.split()) > 1 and sen.strip()]  # Select the word tag

    return words, tags


# Model 1
def model1(train_path, dev_path):

    X_train, Y_train_binary, X_dev, Y_dev_binary = embedding(train_path, dev_path)

    # model = LogisticRegression(max_iter=200)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train_binary)
    Y_pred = model.predict(X_dev)

    f1 = f1_score(Y_dev_binary, Y_pred)
    print(f"Validation F1 Score: {f1}")


# Model 2
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Définir la couche d'entrée (200 neurones, correspondant à la taille des embeddings GloVe)
        self.fc1 = nn.Linear(200, 128)  # Première couche cachée de 128 neurones
        self.fc2 = nn.Linear(128, 64)   # Deuxième couche cachée de 64 neurones
        self.fc3 = nn.Linear(64, 32)    # Ajout d'une troisième couche cachée de 32 neurones
        self.fc4 = nn.Linear(32, 2)    # Couche de sortie de 13 neurones, une pour chaque classe

    def forward(self, x):
        # Forward pass à travers les couches
        x = F.relu(self.fc1(x))  # Activation ReLU pour la première couche cachée
        x = F.relu(self.fc2(x))  # Activation ReLU pour la deuxième couche cachée
        x = F.relu(self.fc3(x))  # Activation ReLU pour la troisième couche cachée
        x = self.fc4(x)          # Pas d'activation ici, cela dépend de votre fonction de perte
        return x


def model2(train_path, dev_path, test_path):

    X_train, Y_train_binary, X_dev, Y_dev_binary = embedding(train_path, dev_path)

    model = SimpleNN()
    learning_rate = 0.001
    num_epochs = 7
    batch_size = 32

    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Conversion des données en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_binary, dtype=torch.long)
    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
    Y_dev_tensor = torch.tensor(Y_dev_binary, dtype=torch.long)

    # Création de TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    dev_dataset = TensorDataset(X_dev_tensor, Y_dev_tensor)

    # Création des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    dev_losses = []
    train_accuracy = []
    dev_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for X_batch, Y_batch in train_loader:
            if torch.cuda.is_available():
                X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += Y_batch.size(0)
            correct_train += (predicted == Y_batch).sum().item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy.append(100 * correct_train / total_train)

        model.eval()
        running_loss = 0.0
        correct_dev = 0
        total_dev = 0
        with torch.no_grad():
            for X_batch, Y_batch in dev_loader:
                if torch.cuda.is_available():
                    X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_dev += Y_batch.size(0)
                correct_dev += (predicted == Y_batch).sum().item()

        dev_loss = running_loss / len(dev_loader)
        dev_losses.append(dev_loss)
        dev_accuracy.append(100 * correct_dev / total_dev)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy[-1]:.2f}%, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy[-1]:.2f}%')

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, Y in dev_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(Y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    f1_binary = f1_score(y_true, y_pred)

    print(f'Binary F1 Score: {f1_binary:.2f}')


# Model 3
class ReviewsDataSet(Dataset, ABC):
    def __init__(self, sentences, sentences_lens, y):
        self.X = sentences
        self.X_lens = sentences_lens
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.X_lens[item], self.y[item]


def tokenize(x_train, x_val):
    word2idx = {"[PAD]": 0, "[UNK]": 1}
    idx2word = ["[PAD]", "[UNK]"]
    for sent in x_train:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([word2idx[word] for word in sent])
    for sent in x_val:
        final_list_test.append([word2idx[word] if word in word2idx else word2idx["[UNK]"] for word in sent])
    return final_list_train, final_list_test, word2idx, idx2word


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features


class MyNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=30, hidden_dim=50, tag_dim=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.hidden2tag = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.hidden_dim, tag_dim))
        self.loss_fn = nn.NLLLoss()

    def forward(self, sentence, sentence_len, tags=None):
        embeds = self.word_embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), -1, self.embedding_dim))
        tag_space = self.hidden2tag(lstm_out[range(len(sentence)), sentence_len - 1, :])
        tag_score = F.softmax(tag_space, dim=1)
        if tags is None:
            return tag_score, None
        loss = self.loss_fn(tag_score, tags)
        return tag_score, loss


def train(model, device, optimizer, train_dataset, val_dataset):
    accuracies = []
    for phase in ["train", "validation"]:
        if phase == "train":
            model.train(True)
        else:
            model.train(False) #or model.evel()
        correct = 0.0
        count = 0
        accuracy = None
        dataset = train_dataset if phase == "train" else val_dataset
        t_bar = tqdm(dataset)
        for sentence, lens, tags in t_bar:
            if phase == "train":
                tag_scores, loss = model(sentence.to(device), lens.to(device), tags.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    tag_scores, _ = model(sentence.to(device), lens.to(device), tags.to(device))
            correct += (tag_scores.argmax(1).to("cpu") == tags).sum()
            count += len(tags)
            accuracy = correct/count
            t_bar.set_description(f"{phase} accuracy: {accuracy:.2f}")
        accuracies += [accuracy]
    return accuracies


def model3(train_path, dev_path, test_path):

    X_train, Y_train_binary, X_dev, Y_dev_binary = embedding(train_path, dev_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    n_classes = max(Y_dev_binary) + 1
    print("n_classes:", n_classes)

    X_train, X_dev, word2idx, idx2word = tokenize(X_train, X_dev)
    vocab_size = len(word2idx)
    print("vocab_size:", vocab_size)

    train_sentence_lens = [min(len(s), 500) for s in X_train]
    test_sentence_lens = [min(len(s), 500) for s in X_dev]

    x_train_pad = padding_(X_train, 500)
    x_test_pad = padding_(X_dev, 500)

    print(x_train_pad.shape, x_test_pad.shape)

    train_dataset = ReviewsDataSet(x_train_pad, train_sentence_lens, Y_train_binary)
    test_dataset = ReviewsDataSet(x_test_pad, test_sentence_lens, Y_dev_binary)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = MyNet(vocab_size, tag_dim=n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)

    best_accuracy = 0
    best_epoch = None
    for epoch in range(7):
        print(f"\n -- Epoch {epoch} --")
        train_accuracy, val_accuracy = train(model, device, optimizer, train_dataloader, test_dataloader)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
        if epoch - best_epoch == 3:
            break
    print(f"best accuracy: {best_accuracy:.2f} in epoch {best_epoch}")


def main():
    train_path = 'data/train.tagged'
    dev_path = 'data/dev.tagged'
    test_path = 'data/test.untagged'

    # model1(train_path, dev_path)
    # model2(train_path, dev_path, test_path)
    model3(train_path, dev_path, test_path)


if __name__ == '__main__':
    main()
