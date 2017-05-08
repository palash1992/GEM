from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

class TopKRanker(oneVr):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction

def evaluateNodeClassification(X, Y, test_ratio):
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_ratio)
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr())
    classif2.fit(X_train, Y_train)        
    prediction = classif2.predict(X_test, top_k_list)
    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')
    return (micro, macro)
