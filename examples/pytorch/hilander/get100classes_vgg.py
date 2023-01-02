from collections import Counter
import pickle
import pandas as pd
import numpy as np
import random

df = pd.read_csv('/content/drive/MyDrive/KLTN_data/VGG-Face2/data/vggface2_test.csv',index_col=0)
labels = df['label'].unique()
img_names = df['img_name']
img_fols = df['original']

#random get 8-15 images for every classes
print(labels)