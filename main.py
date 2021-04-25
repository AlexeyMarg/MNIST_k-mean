import  k_mean

if __name__ == '__main__':
    train_set_path = 'dataset/mnist_train.csv'
    test_set_path = 'dataset/mnist_test.csv'

    classifier = k_mean.k_mean(10, 28 * 28, 255)
    classifier.test(train_set_path, test_set_path)