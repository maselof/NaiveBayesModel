import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from NaiveBayesModel import NaiveBayesModel


def main():
    df = pd.read_csv("src/spamdb.csv", encoding="ISO-8859-1", usecols=["class", "text"])
    X = df['text']
    y = df['class']
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.8,
                                                        shuffle=False)
    NB = NaiveBayesModel()
    NB.init_self_data(X_train, y_train)
    y_pred = NB.predict(X_test)
    count_access = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count_access += 1
    accur = count_access / len(y_test)

    print(accur)


if __name__ == '__main__':
    main()
