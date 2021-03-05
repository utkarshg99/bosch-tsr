import pandas as pd

data = pd.read_csv("gtsrb_kaggle.csv", dtype={"ClassId": int, "ClassIdActual": int})
df = pd.DataFrame(data, columns=['ClassId', 'ClassIdActual'])

pred = df['ClassId']
expc = df['ClassIdActual']

corr = 0
wrng = 0

for i in range(len(pred)):
    if pred[i] == expc[i]:
        corr=corr+1
    else:
        wrng=wrng+1

print(corr)
print(wrng)
print(corr+wrng)
print("Accuracy: {}, Correct: {}, Wrong: {}".format(100.0*corr/(corr+wrng), corr, wrng))