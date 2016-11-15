import pandas
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
with open('data.csv') as f:
    data = pandas.read_csv(f)

values = [value[0] for value in data.values if len(value[0]) > 3]

feature_map = {
    'number': 1,
    'character': 2,
    'space': 3,
    'punctuation': 4,
}

x = []
y = []

sample_size = len(max(values, key=len))

def vectorize(value):
    example = np.zeros(sample_size)
    for i, char in enumerate(value):
        if re.match(r'[a-zA-Z]', char):
            example[i] = feature_map['character']
        elif re.match(r'[0-9]', char):
            example[i] = feature_map['number']
        elif re.match(r'\s', char):
            example[i] = feature_map['space']
        else:
            example[i] = feature_map['punctuation']
    return example

for _ in data.values:
    value = _[0]
    label = _[1]
    example = vectorize(value)
    x.append(example)
    y.append(label)

clf = RandomForestClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf.fit(x_train, y_train)

print(clf.predict([vectorize('1-2 (3)')]))
