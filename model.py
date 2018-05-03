from tabulate import tabulate
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from loading_data import loadingData
from data_vectorizers import count_vectorize, tfidf_vectorize, vectorize_data
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron

#loading data
X_train, y_train = loadingData('train_ref.csv')
X_test, y_test = loadingData('real_val_set.csv')
print('Done loading data...')

#vectorizing data
train_vocab=set([w for twt in X_train for w in twt.split()])
print len(train_vocab)

#counters
x_train_count= count_vectorize(X_train, train_vocab)
x_test_count= count_vectorize(X_test, train_vocab)

# #tfidf
x_train_tfidf= tfidf_vectorize(X_train, train_vocab)
x_test_tfidf= tfidf_vectorize(X_test, train_vocab)

#hashs
x_train= vectorize_data(X_train).transform(X_train)
x_test= vectorize_data(X_train).transform(X_test)
print('Done vectorizing...')

print('begin classification')
classifiers={"multinomial nb": MultinomialNB(),
            "log":  LogisticRegression(multi_class='ovr', solver='sag'),
            'SGD': SGDClassifier(),
            'PT': Perceptron(),
            'PAC': PassiveAggressiveClassifier()}

features={'hashing_vec': [x_train, x_test],
        'counts': [x_train_count, x_test_count],
        'tfidf': [x_train_tfidf, x_test_tfidf],
          }
models_score=[]
for key, fet in features.items():
    for name, cls in classifiers.items():
        print('fitting data')
        cls= cls.fit(fet[0], y_train)
        pred= cls.predict(fet[1])
        emo_label_map = [i.lower() for i in ["ANGER", "DISGUST", "FEAR", "JOY", "SAD", "SURPRISE"]]
        cls_rep= classification_report(y_test, pred, target_names=emo_label_map, digits=2)
        f_score= f1_score(y_test, pred, average='weighted')
        print('classifier name: ' + name + ' and features: ' + str(key))
        models_score.append((name,  f_score))
        print(cls_rep)
print (tabulate(models_score, floatfmt=".4f", headers=("model", 'score')))