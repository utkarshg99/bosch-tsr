import pandas as pd
import numpy as np

classes = 43

data = pd.read_csv("gtsrb_kaggle.csv", dtype={"ClassId": int, "ClassIdActual": int})
df = pd.DataFrame(data, columns=['ClassId', 'ClassIdActual'])

pred = df['ClassId']
expc = df['ClassIdActual']

corr = 0
wrng = 0
conf = np.zeros((classes, classes))
pres = np.zeros(classes)
reca = np.zeros(classes)
f1 = np.zeros(classes) # class-wise


for i in range(len(pred)):
    if pred[i] == expc[i]:
        corr=corr+1
        conf[pred[i]][pred[i]] = conf[pred[i]][pred[i]]+1
    else:
        wrng=wrng+1
        conf[pred[i]][expc[i]] = conf[pred[i]][expc[i]]+1

s1 = np.sum(conf, 0) # col
s2 = np.sum(conf, 1) # row

for i in range(classes):
    pres[i] = conf[i][i]/s2[i]
    reca[i] = conf[i][i]/s1[i]
    f1[i] = 2*pres[i]*reca[i]/(pres[i]+reca[i])
    print("Class: {} F1: {}".format(i+1, f1[i]))

print("Accuracy: {}, Correct: {}, Wrong: {}".format(100.0*corr/(corr+wrng), corr, wrng))
print("Macro F1: {}".format(np.sum(f1)/classes))