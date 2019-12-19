##1.1 data preparation
import os

##data source:http://ai.stanford.edu/~amaas/data/sentiment/.
path = "./aclImdb/test/neg" 
files = os.listdir(path)  
text = []
label = []
for file in files: #遍历文件夹
    if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
        f = open(path+"/"+file, encoding="utf-8") #打开文件
        iter_f = iter(f) #创建迭代器
        str = ""
        for line in iter_f: #遍历文件，一行行遍历，读取文本
            str = str + line
            text.append(str)#每个文件的文本存到List中
            label.append(0)

path = "./aclImdb/test/pos" 
files = os.listdir(path) 
for file in files: 
    if not os.path.isdir(file):
        f = open(path+"/"+file, encoding="utf-8")
        iter_f = iter(f) 
        str = ""
        for line in iter_f: 
            str = str + line
            text.append(str)
            label.append(1)
            
path = "./aclImdb/train/neg" 
files = os.listdir(path) 
for file in files: 
    if not os.path.isdir(file):
        f = open(path+"/"+file, encoding="utf-8")
        iter_f = iter(f) 
        str = ""
        for line in iter_f: 
            str = str + line
            text.append(str)
            label.append(0)
            
path = "./aclImdb/train/pos" 
files = os.listdir(path) 
for file in files: 
    if not os.path.isdir(file):
        f = open(path+"/"+file, encoding="utf-8")
        iter_f = iter(f) 
        str = ""
        for line in iter_f: 
            str = str + line
            text.append(str)
            label.append(1)
			


##1.2 split data into two parts: train && test						
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    text, label, test_size=0.3, stratify=label, random_state=100)
	
	

##2.1 train a classifier with Naive bayes & CountVectorizer & unigram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vec', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##2.2 train a classifier with Naive bayes & TfidfVectorizer & unigram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf2 = Pipeline([
    ('vec', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
clf2.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf2.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##3.1 train a classifier with Naive bayes & CountVectorizer & bigrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf3 = Pipeline([
    ('vec', CountVectorizer(ngram_range=(2, 2))),
    ('nb', MultinomialNB())
])
clf3.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf3.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##3.2 train a classifier with Naive bayes & TfidfVectorizer & bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf4 = Pipeline([
    ('vec', TfidfVectorizer(ngram_range=(2, 2))),
    ('nb', MultinomialNB())
])
clf4.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf4.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##4.1 train a classifier with Logistic regression & CountVectorizer & unigram
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

clf5 = Pipeline([
    ('vec', CountVectorizer()),
    ('nb', LogisticRegression())
])
clf5.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf5.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##4.2 train a classifier with Logistic regression & TfidfVectorizer & unigram
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

clf6 = Pipeline([
    ('vec', TfidfVectorizer()),
    ('nb', LogisticRegression())
])
clf6.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf6.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")


##After all the model training, this model proves to having the highest prediction accuracy, so save this model for later use.
from sklearn.externals import joblib
joblib.dump(clf6, 'model.pkl') 




##5.1 train a classifier with Logistic regression & CountVectorizer & bigrams
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

clf7 = Pipeline([
    ('vec', CountVectorizer(ngram_range=(2, 2))),
    ('nb', LogisticRegression())
])
clf7.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf7.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##5.2 train a classifier with Logistic regression & TfidfVectorizer & bigrams
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

clf8 = Pipeline([
    ('vec', TfidfVectorizer(ngram_range=(2, 2))),
    ('nb', LogisticRegression())
])
clf8.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import classification_report

y_pred = clf8.predict(X_test)
print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

print("####################################################")




##6 train a fasttext classifier 
##data preprocess
def writeData(line):
    f = open("./aclImdb/train.txt", 'a', encoding='utf-8')
    f.writelines(line+'\n')
    f.close()
    
for i in range(len(X_train)):
    if y_train[i] == 0:
        str = "__label__neg" + " " + X_train[i]
        writeData(str)
    else:
        str = "__label__pos" + " " + X_train[i]
        writeData(str)

		
def write_data(line):
    f = open("./aclImdb/test.txt", 'a', encoding='utf-8')
    f.writelines(line+'\n')
    f.close()
    
for i in range(len(X_test)):
    if y_test[i] == 0:
        str = "__label__neg" + " " + X_test[i]
        write_data(str)
    else:
        str = "__label__pos" + " " + X_test[i]
        write_data(str)
		

##model train 
from fasttext import train_supervised
model = train_supervised(
        input = "./aclImdb/train.txt",
        epoch = 25,
        lr = 1.0,
        wordNgrams = 2,
        verbose = 2,
        minCount = 1
)
model.save_model("model.bin")

##model evaluation
from fasttext import load_model

model = load_model("model.bin")
result = model.test("./aclImdb/test.txt")
print(result)