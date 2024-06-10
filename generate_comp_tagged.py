import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from abc import ABC
from tqdm import tqdm


def embedding(train_path, dev_path, test_path):
    train_words, train_tags = preprocess(train_path)
    dev_words, dev_tags = preprocess(dev_path)
    test_words, nb_word = preprocess(test_path, True)

    X_train = train_words + dev_words
    Y_train = train_tags + dev_tags
    X_test = test_words

    return X_train, Y_train, X_test, nb_word


def preprocess(path, condition=False):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not condition:
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

        return words, tags

    else:
        nb_word = 0
        words = []
        sentences = []
        sen = []
        for line in lines:
            if line != "\t\n" and line != "\n":
                new_line = line[:-1]
                sen.append(new_line)
                nb_word += 1
            else:
                sentences.append(sen)
                sen = []

        for sen in sentences:
            word_line = []
            for line in sen:
                if line.strip():
                    word_line.append(line.lower())
            words.append(word_line)

        return words, nb_word


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
    f1 = 0
    accuracies = []
    y_true = []
    y_pred = []
    for phase in ["train"]:#, "validation"]:
        # if phase == "train":
        model.train(True)
        # else:
        #     model.train(False)
        accuracy = None
        # dataset = train_dataset if phase == "train" else val_dataset
        t_bar = tqdm(train_dataset)
        for sentence, lens, tags in t_bar:
            sentence, lens, tags = sentence.to(device), lens.to(device), tags.to(device).long()
            # if phase == "train":
            model.zero_grad()
            tag_scores, loss = model(sentence, lens, tags)
            loss.backward()
            optimizer.step()
            # else:
            #     with torch.no_grad():
            #         tag_scores, _ = model(sentence, lens, tags)

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

        f1 = f1_score(y_true, y_pred)
        print(f'F1 Score for {phase} phase: {f1:.4f}')
        accuracies += [accuracy]

    return f1


def model_comp(train_path, dev_path, test_path):
    seed = 27
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, Y_train_binary, X_test, nb_word = embedding(train_path, dev_path, test_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_classes = 2

    X_train, X_test, word2idx, idx2word = tokenize(X_train, X_test)
    vocab_size = len(word2idx)

    train_sentence_lens = [min(len(s), 200) for s in X_train]
    test_sentence_lens = [min(len(s), 200) for s in X_test]

    x_train_pad = padding_(X_train, 200)
    Y_train_binary = padding_(Y_train_binary, 200)
    x_test_pad = padding_(X_test, 200)
    Y_test = np.zeros((len(x_test_pad), 200), dtype=int)

    train_dataset = ReviewsDataSet(x_train_pad, train_sentence_lens, Y_train_binary)
    test_dataset = ReviewsDataSet(x_test_pad, test_sentence_lens, Y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = MyNet(vocab_size, tag_dim=n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)

    output_path = "comp_931202543_932191265.tagged"
    for epoch in range(30):
        print(f"\n -- Epoch {epoch} --")
        f1 = train(model, device, optimizer, train_dataloader, test_dataloader)

        if f1 > 0.99:
            break

    y_pred = []
    model.eval()
    for sentence, lens, tags in test_dataloader:
        sentence, lens, tags = sentence.to(device), lens.to(device), tags.to(device).long()
        with torch.no_grad():
            tag_scores, _ = model(sentence, lens, tags)
        predicted = tag_scores.argmax(dim=2)
        predicted_cpu = predicted.cpu()
        for i, tag in enumerate(predicted_cpu):
            new_tag = tag.numpy()[:lens[i]]
            y_pred.extend(new_tag)

    with open(output_path, 'w'):
        pass

    with open(test_path, 'r', encoding='utf-8') as f_tst, open(output_path, 'w', encoding='utf-8') as f_tag:
        i = 0
        for line in f_tst:
            if line.strip():
                word = line.strip()
                if y_pred[i] == 0 or len(word) == 1 or word == 'gt' or word == 'the' or word == 'The':
                    tag = 'O'
                else:
                    tag = '1'
                f_tag.write(f"{word}\t{tag}\n")
                i += 1
            else:
                f_tag.write("\n")


def main():
    train_path = 'data/train.tagged'
    dev_path = 'data/dev.tagged'
    test_path = 'data/test.untagged'

    model_comp(train_path, dev_path, test_path)


if __name__ == '__main__':
    main()
