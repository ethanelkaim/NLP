import numpy as np
from preprocessing import read_test
from tqdm import tqdm


def q(v, u, t, w, k):
    numerator = 0
    denominator = 0
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
    bp_matrix = np.zeros((num_words + 1, num_tags, num_tags), dtype=int)
    pi_matrix[0, tags.index('*'), tags.index('*')] = 1

    # Iterating over words
    for k in range(1, num_words + 1):
        # Iterating over current tag
        for v, current_tag in enumerate(tags):
            # Iterating over previous tags
            for u, prev_tag in enumerate(tags):
                # Iterating over pre_previous tags
                prob = {tag: 0 for tag in tags}
                for t, pre_pre_tag in enumerate(tags):
                    prob[t] = pi_matrix[(k - 1, t, u)] * q(v, u, t, w, k)

                argmax_t = -1
                max_t = float('-inf')
                for arg_t, t in enumerate(prob):
                    if t > max_t:
                        max_t = t
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
            score = pi_matrix[num_words - 3][u][v]
            if score > max_last_score:
                max_last_score = score
                max_last_tag = v

    best_sequence.append(tags[max_last_tag])

    # Backtrack to find the best sequence of tags
    for k in range(num_words - 3, 0, -1):
        prev_tag_index = bp_matrix[k][max_last_tag][0]
        best_sequence.insert(0, tags[prev_tag_index])
        max_last_tag = prev_tag_index

    return best_sequence


def extract_trigram_features(sentence, i, current_tag, prev_tag, feature2id):
    """
    Extract features for the trigram MEMM model.
    """
    features = []
    # Example: Add features related to current word, current tag, previous word, previous tag, etc.
    features.append((sentence[i], current_tag, sentence[i - 1], prev_tag))
    # Add more features as needed

    # Convert features to feature indices
    feature_indices = [feature2id.get(f, feature2id.get('UNK')) for f in features]

    return feature_indices
#
#
# def calculate_score(features, pre_trained_weights):
#     """
#     Calculate the score for a given set of features using pre-trained weights.
#     """
#     score = 0.0
#     for feature_index in features:
#         score += pre_trained_weights[feature_index]
#     return score


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
