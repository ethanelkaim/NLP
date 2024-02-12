import pickle

from test_file import compare_files
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def main():
    threshold = 5
    lam = 0.5

    for i in range(1):
        train_path = ["data/train1.wtag", "data/train1.wtag", "data/train2.wtag"]
        test_path = ["data/test1.wtag", "data/comp1.words", "data/comp2.words"]

        weights_path = 'weights.pkl'
        predictions_path = ['predictions.wtag', 'comp_m1_931202543_932191265.wtag', 'comp_m2_931202543_932191265.wtag']

        statistics, feature2id = preprocess_train(train_path[i], threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        print(pre_trained_weights)
        tag_all_test(test_path[i], pre_trained_weights, feature2id, predictions_path[i])
        print("\n ***   End of test " + test_path[i] + "   ***\n")

        if i == 0:
            fraction, prob, conf_mat = compare_files(test_path[0], predictions_path[0])
            print(f"fraction : {fraction}\nprob : {prob}\n                         ***   conf_mat   ***\n\n {conf_mat}")


if __name__ == '__main__':
    with open('predictions.wtag', 'w') as file:
        pass
    main()
