import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

#Function to extract the MNIST data set and convert it to a CSV file
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


#Function to calculate the posterior probabilities of the misclassified MNIST digits
def posterior_prob(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

#Function call for conversion
convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
"mnist_train.csv", 60000)

#Storing the extracted images and labels in the form of a matrix named data
data=pd.read_csv("mnist_train.csv").as_matrix()

#Naive Bayes Bernoulli distribution model
clf = BernoulliNB() 

#Training dateset (size=50000)
xtrain=data[0:50000,1:]
train_label=data[0:50000,0]
clf.fit(xtrain,train_label) #training the data using bernoulli distribution from Sklearn package

#The class probabilities
print("The class probabilities are:")
i=0
ynew = clf.predict_proba(xtrain)
for i in range(10):
    print("Digit=%s, Class probailities of the digits: \n%s" % (i, ynew[i]))

#Testing dataset(size=10000)
xtest=data[50000:,1:]
actual_label=data[50000:,0]

p=clf.predict(xtest)

#Creating a confusion matrix to describe the performance of the model
cm=confusion_matrix(actual_label,p)
print("The confusion matrix is:")
print(cm)
pt.imshow(cm)

#Performance metrics of the testing set 
accuracy_score =sklearn.metrics.accuracy_score(actual_label, p, sample_weight=None)
print("\nThe testing set accuracy is:", accuracy_score) 
print("The error rate of the MNIST testing set is:", 1 - accuracy_score) #error rate of MNIST test set

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']
print("\nThe classification report of MNIST digits:",classification_report(actual_label, p, target_names=target_names)) #classification report using the confusion matrix


#The five misclassified digits and their respective posterior probabilities 
err = 0
index = [0,0,0,0,0,0,0,0,0,0]
actual = [0,0,0,0,0,0,0,0,0,0]

print("\nThe five misclassified digits and their respective posterior probabilities are below:")
for i in range(0,10):
    index[i] = posterior_prob(train_label, lambda x: x == i)
    actual[i] = (len(index[i])/50000)

for i in range(0,9999):
    if p[i]==actual_label[i]:
        continue
    else:
        if err < 5:
            err += 1
            d=xtest[i]
            d.shape=(28,28)
            pt.imshow(d,cmap='gray')
            print("Predicted value:", p[i], "\t\tCorrect Value:", actual_label[i], "\nPosterior probability:", actual)
            print("\n")
            pt.show()









