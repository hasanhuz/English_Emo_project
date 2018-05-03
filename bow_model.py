from tabulate import tabulate
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from loading_data import loadingData
from bow_means import MeanEmbeddingVectorizer, loadW2V
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

#loading data
X_train, y_train = loadingData('train_ref.csv')
X_test, y_test = loadingData('real_val_set.csv')
print('Done loading data...')

#loading w2v model
w2v_model= loadW2V('w2v_emedding.csv', X_train)

print(len(w2v_model))
svm_w2v = Pipeline([("w2v_vectorizer", MeanEmbeddingVectorizer(w2v_model)),
                    ("w2v_svm", SVC(kernel = 'linear'))])

log_w2v = Pipeline([("w2v_vectorizer", MeanEmbeddingVectorizer(w2v_model)),
                    ("log",  LogisticRegression(multi_class='ovr', solver='sag', random_state=42))])

classifiers={"svm_w2v": svm_w2v,
            "log_w2v": log_w2v}

#  accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test)
models_score=[]
for name, cls in classifiers.items():
    print('fitting data')
    cls= cls.fit(X_train, y_train)
    pred= cls.predict(X_test)
    emo_label_map = [i.lower() for i in ["ANGER", "DISGUST", "FEAR", "JOY", "SAD", "SURPRISE"]]
    cls_rep= classification_report(y_test, pred, target_names=emo_label_map, digits=2)
    f_score= f1_score(y_test, pred, average='weighted')
    print 'classifier name: ' + name
    models_score.append((name,  f_score))
    print cls_rep
print tabulate(models_score, floatfmt=".4f", headers=("model", 'score'))
print('...Done...')