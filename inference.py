from preprocessing import read_test
from tqdm import tqdm


# Define a function to calculate the score of a tag transition
def score_transition(prev_tag, curr_tag):
    # Your score calculation logic here
    return 0.0  # Placeholder value


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
        Write your MEMM Viterbi implementation below
        You can implement Beam Search to improve runtime
        Implement q efficiently (refer to conditional probability definition in MEMM slides)
        """

    # Initialize variables
    n_words = len(sentence)
    tags = list(feature2id.feature_statistics.tags)  # List of possible tags

    # Initialize data structures for Viterbi algorithm
    best_scores = [{tag: 0.0 for tag in tags} for _ in range(n_words)]
    back_pointers = [{} for _ in range(n_words)]

    # Perform Viterbi algorithm
    for i, word in enumerate(sentence):
        if i == 0:
            prev_best_scores = {"*": 0.0}  # Start symbol
        else:
            prev_best_scores = best_scores[i - 1]

        for curr_tag in tags:
            max_score = float('-inf')
            best_prev_tag = None

            # Calculate the score for transitioning to the current tag
            transition_score = score_transition("*", curr_tag)  # Example: Initial transition from start symbol

            for prev_tag, prev_score in prev_best_scores.items():
                # Calculate the score for transitioning from the previous tag to the current tag
                # and adding the score of the current word-tag pair
                emission_score = calculate_emission_score(word, curr_tag, pre_trained_weights, feature2id)
                score = prev_score + transition_score + emission_score

                if score > max_score:
                    max_score = score
                    best_prev_tag = prev_tag

            best_scores[i][curr_tag] = max_score
            back_pointers[i][curr_tag] = best_prev_tag

    # Perform backtracking to find the best sequence of tags
    best_sequence = []
    max_final_score = float('-inf')
    best_final_tag = None

    for tag, score in best_scores[-1].items():
        if score > max_final_score:
            max_final_score = score
            best_final_tag = tag

    current_tag = best_final_tag
    for bp in reversed(back_pointers):
        best_sequence.insert(0, current_tag)
        current_tag = bp[current_tag]

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
