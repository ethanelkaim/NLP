import heapq
import numpy as np
from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import math


def q(w, features_dict, pre_trained_weights, tags, total_score=0):

    index_vector = represent_input_with_features(w, features_dict)
    for i in index_vector:
        total_score += pre_trained_weights[i]
    numerator = math.exp(total_score)

    denominator = 0
    all_w = []
    for tag in tags:
        all_w.append((w[0], tag, w[2], w[3], w[4], w[5], w[6]))

    for w in all_w:
        score = 0
        index_vector = represent_input_with_features(w, features_dict)
        for j in index_vector:
            score += pre_trained_weights[j]
        denominator += math.exp(score)

    return numerator / denominator


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
        Write your MEMM Viterbi implementation below
        You can implement Beam Search to improve runtime
        Implement q efficiently (refer to conditional probability definition in MEMM slides)
        """
    # Extract relevant information
    tags = list(feature2id.feature_statistics.tags) + ["*"]
    num_tags = len(tags)
    num_words = len(sentence)

    pi_matrix = np.zeros((num_words + 1, num_tags, num_tags))
    bp_matrix = np.empty((num_words + 1, num_tags, num_tags), dtype=object)
    pi_matrix[0, tags.index('*'), tags.index('*')] = 1

    # Iterating over words
    for k in range(1, num_words):
        if k == num_words - 1:
            next_word = '~'
        else:
            next_word = sentence[k + 1]
        previous_word = sentence[k - 1]
        if k == 1:
            pre_previous_word = '*'
        else:
            pre_previous_word = sentence[k - 2]

        # Beam search of the best two tags for previous and pre-previous positions
        if k == 1:
            top_prev_tags = [tags.index('*')]
            top_pre_prev_tags = [tags.index('*')]
        elif k == 2:
            top_prev_tags = np.argsort(-pi_matrix[k - 1, :, :].max(axis=0))[:2]
            top_pre_prev_tags = [tags.index('*')]
        else:
            top_prev_tags = np.argsort(-pi_matrix[k - 1, :, :].max(axis=0))[:2]
            top_pre_prev_tags = np.argsort(-pi_matrix[k - 2, :, :].max(axis=0))[:2]

            # Iterating over current tag
        for v, current_tag in enumerate(tags):
            # Iterate over the best two previous tags and pre-previous tags
            for u in top_prev_tags:
                for t in top_pre_prev_tags:
                    w = [sentence[k], current_tag, previous_word, tags[u], pre_previous_word, tags[t], next_word]
                    prob = pi_matrix[k - 1, t, u] * q(w, feature2id.feature_to_idx, pre_trained_weights, tags)

                    # Update pi_matrix and bp_matrix
                    if prob > pi_matrix[k, u, v]:
                        pi_matrix[k, u, v] = prob
                        bp_matrix[k, u, v] = tags[t]

    # Backtrack to find the best tag sequence
    best_sequence = []
    max_last_score = float('-inf')
    max_last_tag = -1
    max_last_last_tag = -1

    # Find the maximum probability for the last word
    for u, prev_tag in enumerate(tags):
        for v, current_tag in enumerate(tags):
            score = pi_matrix[num_words - 2, u, v]
            if score > max_last_score:
                max_last_score = score
                max_last_tag = v
                max_last_last_tag = u

    best_sequence.append(tags[max_last_last_tag])
    best_sequence.append(tags[max_last_tag])

    # Backtrack to find the best sequence of tags
    for k in range(num_words - 2, 2, -1):
        prev_tag = bp_matrix[k, max_last_last_tag, max_last_tag]
        best_sequence.insert(0, prev_tag)
        max_last_tag = max_last_last_tag
        max_last_last_tag = tags.index(prev_tag)

    return best_sequence


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    count = 0

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
        count += 1
    output_file.close()
