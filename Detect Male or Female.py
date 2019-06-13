import pandas as pd
import numpy as np
from sklearn import tree

# making one list of Genders (Male/Female)

data = pd.read_csv("gender_data.csv")

gender = np.array(data.Gender)
gender = gender.tolist()

#converting the all columns into individual list

l1 = np.array(data.Height)
l1 = l1.tolist()
l2 = np.array(data.Weight)
l2 = l2.tolist()
l3 = np.array(data.Index)
l3 = l3.tolist()


measure = np.column_stack((l1,l2,l3))
measure = measure.tolist()



f=tree.DecisionTreeClassifier()
f=f.fit(measure,gender)
print ("Enter Your Height")
x=int(input(">>"))
print ("Enter Your Weight")
y=int(input(">>"))
print ("Enter Your Index")
z=int(input(">>"))

prediction = f.predict([[x,y,z]])

if prediction[0] == "Male":
    print("The Person is Male")
else :
    print("The Person is female")