from collections import Counter
import pickle
import pandas as pd
import numpy as np
import random

#train_path = 'data/subcenter_arcface_deepglint_train_1_in_10_recreated.pkl'
train_path = 'data/subcenter_arcface_deepglint_imdb_features.pkl'
with open(train_path, "rb") as f:
    features, labels = pickle.load(f)

a = Counter(labels)
#sorted(a.items(), key=lambda x: x[1], reverse=True)
#print(a.most_common(10))

feature1, labels1 = [],[]
#get 10 image for every 100 classes
for l in a.most_common(100):
    temp_label = []
    temp_feature = []
    ran = random.randint(30,50)
    #print(ran)
    for i in range(len(labels)):
        if l[0] == labels[i] and len(temp_label) < ran:
            #print('found')
            temp_label.append(labels[i])
            temp_feature.append(features[i])
    feature1.extend(temp_feature)
    labels1.extend(temp_label)
        

print(len(feature1))
print(len(labels1))
data = [np.array(feature1), np.array(labels1)]

with open('imdb_100classes_random30to50.pkl', 'wb') as f:
    pickle.dump(data, f)