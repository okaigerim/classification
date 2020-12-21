from sklearn import tree

clf = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
#Measurements: height - cm, weight - kg, shoe size - RU
X = [[180,65,42], [187,70,42], [172, 55, 38], [165, 45,35],
        [185, 70, 43], [184,74,44], [160, 46, 38], [167, 49, 37], [168, 47, 37],
        [190, 74, 42], [180, 65, 41]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Train the model
clf = clf.fit(X, Y)

result_1 = clf.predict([[190, 70, 43]])
result_2 = clf.predict([[185, 74, 41]])
result_3 = clf.predict([[160, 47, 36]])


print("1-{}, 2-{}, 3-{}".format(result_1, result_2, result_3))
