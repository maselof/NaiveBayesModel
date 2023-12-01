import re
from collections import defaultdict


class NaiveBayesModel:
    def __init__(self):
        self.pspam = None
        self.pham = None
        self.NEUTRAL_WORDS = {
        "and",
        "the",
        "in",
        "to",
        "of",
        "a",
        "an",
        "is",
        "it",
        "on",
        "with",
        "for",
        }
        self.words_spam = None
        self.words_ham = None

    def init_self_data(self, X, y):
        count_spam, count_ham = 0, 0
        for i in range(len(y)):
            if y[i] == 'spam':
                count_spam += 1
            elif y[i] == 'ham':
                count_ham += 1
        self.pspam = count_spam / len(y)
        self.pham = count_ham / len(y)
        self.words_spam = {}
        self.words_ham = {}

        for i in range(len(X)):
            words = self.get_words(X[i])
            words = [w for w in words if w not in self.NEUTRAL_WORDS]

            for w in words:
                if y[i] == 'spam':
                    if w not in self.words_spam:
                        self.words_spam[w] = 0
                    self.words_spam[w] += 1
                else:
                    if w not in self.words_ham:
                        self.words_ham[w] = 0
                    self.words_ham[w] += 1

        a = 1

        for words in self.words_spam.keys():
            self.words_spam[words] = (a + len(self.words_spam)) / (a * len(X) + self.words_spam[words])

        for words in self.words_ham.keys():
            self.words_ham[words] = (a + len(self.words_ham)) / (a * len(X) + self.words_ham[words])

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            pspam = self.pspam
            pham = self.pham
            words = self.get_words(X[i])
            words = [w for w in words if w not in self.NEUTRAL_WORDS]

            for w in words:
                if w in self.words_spam:
                    pspam *= self.words_spam[w]
                if w in self.words_ham:
                    pham *= self.words_ham[w]

            if pspam > pham:
                y_pred.append('spam')
            else:
                y_pred.append('ham')
        return y_pred




    @staticmethod
    def get_words(text):
        return re.findall(r"\b\w+\b", text.lower())


