import math
import json
import pandas as pd
import numpy as np

def execute(classes=48, filename="output.csv"):

    data = pd.read_csv(filename, dtype={"ClassId": int, "ClassIdActual": int})
    df = pd.DataFrame(data, columns=['ClassId', 'ClassIdActual'])

    pred = df['ClassId']
    expc = df['ClassIdActual']

    corr = 0
    wrng = 0
    conf = np.zeros((classes, classes))
    prec = np.zeros(classes) # precision
    reca = np.zeros(classes) # recall/sensitivity/true positive rate
    spec = np.zeros(classes) # specitivity/true negative rate
    posl = np.zeros(classes) # positive likelihood
    negl = np.zeros(classes) # negative likelihood
    bcrt = np.zeros(classes) # balanced classification rate
    bert = np.zeros(classes) # balance error rate/half total error rate
    matc = np.zeros(classes) # Matthew's Correlation

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
        # TP: conf[i][i]
        # TN: corr-conf[i][i]
        # FP: s2[i]-conf[i][i]
        # FN: s1[i]-conf[i][i]
        prec[i] = conf[i][i]/s2[i]
        reca[i] = conf[i][i]/s1[i]
        spec[i] = (corr-conf[i][i])/(s2[i]+corr-2*conf[i][i])
        posl[i] = spec[i]/(1-spec[i])
        negl[i] = (1-spec[i])/spec[i]
        bcrt[i] = 0.5*(reca[i]+spec[i])
        bert[i] = 1-bcrt[i]
        matc[i] = (conf[i][i]*(corr-conf[i][i]) - (s2[i]-conf[i][i])*(s1[i]-conf[i][i]))/math.sqrt((s2[i])*(s1[i])*(corr+s2[i]-2*conf[i][i])*(corr+s1[i]-2*conf[i][i]))
        f1[i] = 2*prec[i]*reca[i]/(prec[i]+reca[i]) if prec[i]+reca[i] != 0 else 0
        if math.isnan(f1[i]): f1[i]=0
        print("Class: {} F1: {} Acc: {}".format(i, f1[i], conf[i][i]/s1[i]))

    print("Accuracy: {}, Correct: {}, Wrong: {}".format(100.0*corr/(corr+wrng), corr, wrng))
    print("Macro F1: {}".format(np.mean(f1)))
    res_dict = {
        "prec": prec.tolist(),
        "reca": reca.tolist(),
        "spec": spec.tolist(),
        "posl": posl.tolist(),
        "negl": negl.tolist(),
        "bcrt": bcrt.tolist(),
        "bert": bert.tolist(),
        "matc": matc.tolist(),
        "f1": f1.tolist(),
        "Mprec": np.mean(prec),
        "Mreca": np.mean(reca),
        "Mspec": np.mean(spec),
        "Mposl": np.mean(posl),
        "Mnegl": np.mean(negl),
        "Mbcrt": np.mean(bcrt),
        "Mbert": np.mean(bert),
        "Mmatc": np.mean(matc),
        "Mf1": np.mean(f1),
        "nclasses": classes,
        "accuracy": 100.0*corr/(corr+wrng),
        "ncorrect": corr,
        "nwrong": wrng
    }

    with open("result.json", "w") as outfile:  
        json.dump(res_dict, outfile, indent=4)

if __name__ == "__main__":
    execute()