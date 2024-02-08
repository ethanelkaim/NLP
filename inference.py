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
    for k in tqdm(range(1, num_words), total=num_words - 1):
        if k == num_words - 1:
            next_word = '~'
        else:
            next_word = sentence[k + 1]
        previous_word = sentence[k - 1]
        pre_previous_word = sentence[k - 2]

        # Iterating over current tag
        for v, current_tag in enumerate(tags):
            # Iterating over previous tags
            for u, prev_tag in enumerate(tags):
                # Iterating over pre_previous tags
                prob = {tag: 0 for tag in tags}
                for t, pre_pre_tag in enumerate(tags):
                    w = [sentence[k], current_tag, previous_word, prev_tag, pre_previous_word, pre_pre_tag, next_word]
                    # qu = q(w, feature2id.feature_to_idx, pre_trained_weights, tags)
                    # pim = pi_matrix[(k - 1, t, u)]
                    # prob[pre_pre_tag] = pim * qu
                    prob[pre_pre_tag] = pi_matrix[(k - 1, t, u)] * q(w, feature2id.feature_to_idx, pre_trained_weights, tags)

                argmax_t = -1
                max_t = float('-inf')
                for arg_t, p_t in prob.items():
                    if p_t > max_t:
                        max_t = p_t
                        argmax_t = arg_t

                pi_matrix[k, u, v] = max_t
                bp_matrix[k, u, v] = argmax_t

    # Backtrack to find the best tag sequence
    best_sequence = []
    max_last_score = float('-inf')
    max_last_tag = -1

    # Find the maximum probability for the last word
    for u, prev_tag in enumerate(tags):
        for v, current_tag in enumerate(tags):
            score = pi_matrix[num_words - 2, u, v]
            if score > max_last_score:
                max_last_score = score
                max_last_tag = v

    best_sequence.append(tags[max_last_tag])

    # Backtrack to find the best sequence of tags
    for k in range(num_words - 2, 0, -1):
        prev_tag_index = bp_matrix[k, max_last_tag]
        if isinstance(prev_tag_index, np.ndarray):
            prev_tag_index = prev_tag_index.item()
        best_sequence.insert(0, tags[prev_tag_index])
        max_last_tag = prev_tag_index

    return best_sequence


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
