from preprocessing import preprocess_train
from optimization import get_optimal_vector


def main():
    threshold = 1
    lam = 1

    for i in range(1):
        train_path = ["data/train1.wtag", "data/train2.wtag"]
        weights_path = ['weights1.pkl', 'weights2.pkl']
        predictions_path = ['comp_m1_931202543_932191265.wtag', 'comp_m2_931202543_932191265.wtag']

        with open(predictions_path[i], 'w'):
            pass

        statistics, feature2id = preprocess_train(train_path[i], threshold)

        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path[i], lam=lam)


if __name__ == '__main__':
    main()
