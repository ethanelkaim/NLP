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
from tqdm import tqdm

GLOVE_PATH = 'glove-twitter-200'


def embedding(train_path, dev_path, condition=False):
    if not condition:
        try:
            glove = KeyedVectors.load('glove_model.pkl')
        except FileNotFoundError:
            glove = downloader.load(GLOVE_PATH)
            with open('glove_model.pkl', 'wb') as f:
                pickle.dump(glove, f)

    train_words, train_tags = preprocess(train_path, condition)
    dev_words, dev_tags = preprocess(dev_path, condition)

    X_train, Y_train = [], []
    X_dev, Y_dev = [], []

    if not condition:
        for word, tag in zip(train_words, train_tags):
            if word in glove.key_to_index:
                X_train.append(glove[word])
                Y_train.append(tag)

        for word, tag in zip(dev_words, dev_tags):
            if word in glove.key_to_index:
                X_dev.append(glove[word])
                Y_dev.append(tag)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_dev = np.array(X_dev)
        Y_dev = np.array(Y_dev)

        Y_train = [0 if tag == 'O' else 1 for tag in Y_train]
        Y_dev = [0 if tag == 'O' else 1 for tag in Y_dev]

    else:
        X_train = train_words
        Y_train = train_tags
        X_dev = dev_words
        Y_dev = dev_tags

    return X_train, Y_train, X_dev, Y_dev


def preprocess(path, condition=True):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if condition:
        words, tags = [], []
        sentences = []
        sen = []
        for line in lines:
            if line != "\t\n" and line != "\n":
                sen.append(line)
            else:
                sentences.append(sen)
                sen = []

        for sen in sentences:
            word_line = []
            tag_line = []
            for line in sen:
                if line.strip():
                    word_line.append(line.split()[0].lower())
                if len(line.split()) > 1 and line.strip():
                    if line.split()[1] == 'O':
                        tag_line.append(0)
                    else:
                        tag_line.append(1)
            words.append(word_line)
            tags.append(tag_line)

    else:
        words = [sen.split()[0].lower() for sen in lines if sen.strip()]  # Select first word, make lowercase
        tags = [sen.split()[1] for sen in lines if len(sen.split()) > 1 and sen.strip()]  # Select the word tag

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
        self.fc2 = nn.Linear(128, 64)  # Deuxième couche cachée de 64 neurones
        self.fc3 = nn.Linear(64, 32)  # Ajout d'une troisième couche cachée de 32 neurones
        self.fc4 = nn.Linear(32, 2)  # Couche de sortie de 13 neurones, une pour chaque classe

    def forward(self, x):
        # Forward pass à travers les couches
        x = F.relu(self.fc1(x))  # Activation ReLU pour la première couche cachée
        x = F.relu(self.fc2(x))  # Activation ReLU pour la deuxième couche cachée
        x = F.relu(self.fc3(x))  # Activation ReLU pour la troisième couche cachée
        x = self.fc4(x)  # Pas d'activation ici, cela dépend de votre fonction de perte
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

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy[-1]:.2f}%, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy[-1]:.2f}%')

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
    word2idx = {"PAD": 0, "UNK": 1}
    idx2word = ["PAD", "UNK"]
    for sent in x_train:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([word2idx[word] for word in sent])
    for sent in x_val:
        final_list_test.append([word2idx[word] if word in word2idx else word2idx['UNK'] for word in sent])
    return final_list_train, final_list_test, word2idx, idx2word


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features


class MyNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=30, hidden_dim=50, tag_dim=2):
        super(MyNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Sequential(nn.ReLU(), nn.Linear(self.hidden_dim, tag_dim))
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, sentence, sentence_lens, tags=None):
        embeds = self.word_embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        if tags is not None:
            loss = self.loss_fn(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))
            return tag_scores, loss
        return tag_scores, None


def train(model, device, optimizer, train_dataset, val_dataset):
    accuracies = []
    y_true = []
    y_pred = []
    for phase in ["train", "validation"]:
        if phase == "train":
            model.train(True)
        else:
            model.train(False)
        correct = 0.0
        count = 0
        accuracy = None
        dataset = train_dataset if phase == "train" else val_dataset
        t_bar = tqdm(dataset)
        for sentence, lens, tags in t_bar:
            sentence, lens, tags = sentence.to(device), lens.to(device), tags.to(device).long()
            if phase == "train":
                model.zero_grad()
                tag_scores, loss = model(sentence, lens, tags)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    tag_scores, _ = model(sentence, lens, tags)

            # Convert softmax scores to predictions
            # Assuming dim=2 since your tag_scores are likely [batch, seq_len, n_classes]
            predicted = tag_scores.argmax(dim=2)

            # Move tensors back to CPU for compatibility with NumPy and sklearn
            tags_cpu = tags.cpu()
            predicted_cpu = predicted.cpu()

            # Flatten the tensors to collapse the batch and sequence dimensions
            for tag in tags_cpu:
                y_true.extend(tag.numpy())
            for tag in predicted_cpu:
                y_pred.extend(tag.numpy())

            correct += (tag_scores[0].argmax(1).to(device) == tags).sum()
            count += len(tags)
            accuracy = correct / count
            t_bar.set_description(f"{phase} accuracy: {accuracy:.2f}")

        f1 = f1_score(y_true, y_pred)
        print(f'F1 Score for {phase} phase: {f1:.4f}')
        accuracies += [accuracy]

    return accuracies[0], accuracies[1], f1


def model3(train_path, dev_path, test_path):
    X_train, Y_train_binary, X_dev, Y_dev_binary = embedding(train_path, dev_path, True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    n_classes = 2
    print("n_classes:", n_classes)

    X_train, X_dev, word2idx, idx2word = tokenize(X_train, X_dev)
    vocab_size = len(word2idx)
    print("vocab_size:", vocab_size)

    train_sentence_lens = [min(len(s), 80) for s in X_train]
    test_sentence_lens = [min(len(s), 80) for s in X_dev]

    x_train_pad = padding_(X_train, 80)
    Y_train_binary = padding_(Y_train_binary, 80)
    x_test_pad = padding_(X_dev, 80)
    Y_dev_binary = padding_(Y_dev_binary, 80)

    print(x_train_pad.shape, x_test_pad.shape)

    train_dataset = ReviewsDataSet(x_train_pad, train_sentence_lens, Y_train_binary)
    test_dataset = ReviewsDataSet(x_test_pad, test_sentence_lens, Y_dev_binary)

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    model = MyNet(vocab_size, tag_dim=n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)

    best_f1 = 0
    best_epoch = 0
    for epoch in range(1000):
        print(f"\n -- Epoch {epoch} --")
        train_accuracy, val_accuracy, f1 = train(model, device, optimizer, train_dataloader, test_dataloader)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
        if epoch - best_epoch == 10:
            break
    print(f"\nBest F1 score is : {best_f1:.4f} in epoch {best_epoch}")


def main():
    train_path = 'data/train.tagged'
    dev_path = 'data/dev.tagged'
    test_path = 'data/test.untagged'

    # model1(train_path, dev_path)
    # model2(train_path, dev_path, test_path)
    model3(train_path, dev_path, test_path)


if __name__ == '__main__':
    main()
