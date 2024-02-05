from preprocessing import read_test
from tqdm import tqdm


# Define a function to calculate the score of a tag transition
# def score_transition(prev_tag, curr_tag):
    # Your score calculation logic here

    # return 0.0  # Placeholder value


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
        Write your MEMM Viterbi implementation below
        You can implement Beam Search to improve runtime
        Implement q efficiently (refer to conditional probability definition in MEMM slides)
        """
    # Extract relevant information
    num_tags = len(feature2id.tags)
    num_words = len(sentence)
    tags = list(feature2id.tags)

    # Initialize matrices for probabilities and backpointers
    # dimensions: (num_words - 2) x num_tags x num_tags
    # probabilities[k][i][j] represents the probability of tag j at word k-2 given tags i at k-1 and tags at k
    probabilities = np.zeros((num_words - 2, num_tags, num_tags))
    backpointers = np.zeros((num_words - 2, num_tags, num_tags), dtype=int)

    # Iterating over words
    for i in range(2, num_words):
        # Iterating over current tag
        for k, current_tag in enumerate(tags):
            # Iterating over previous tags
            for j, prev_tag in enumerate(tags):
                # Extract features for trigram
                features = extract_trigram_features(sentence, i, current_tag, prev_tag, feature2id)

                # Calculate the score using pre-trained weights
                score = calculate_score(features, pre_trained_weights)

                # Update probabilities
                if i == 2:  # Base case for first trigram
                    probabilities[0][0][k] = score
                else:
                    # Find the maximum probability and corresponding previous tag
                    max_score = float('-inf')
                    max_prev_tag = -1
                    for prev_tag_index, prev_tag_prob in enumerate(probabilities[i - 3, :, :]):
                        temp_score = prev_tag_prob[j] + score
                        if temp_score > max_score:
                            max_score = temp_score
                            max_prev_tag = prev_tag_index
                    probabilities[i - 2, j, k] = max_score
                    backpointers[i - 2, j, k] = max_prev_tag

    # Backtrack to find the best tag sequence
    best_sequence = []
    max_last_score = float('-inf')
    max_last_tag = -1

    # Find the maximum probability for the last word
    for j, prev_tag in enumerate(tags):
        for k, current_tag in enumerate(tags):
            score = probabilities[num_words - 3][j][k]
            if score > max_last_score:
                max_last_score = score
                max_last_tag = k

    best_sequence.append(tags[max_last_tag])

    # Backtrack to find the best sequence of tags
    for i in range(num_words - 3, 0, -1):
        prev_tag_index = backpointers[i][max_last_tag][0]
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


def calculate_score(features, pre_trained_weights):
    """
    Calculate the score for a given set of features using pre-trained weights.
    """
    score = 0.0
    for feature_index in features:
        score += pre_trained_weights[feature_index]
    return score

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
