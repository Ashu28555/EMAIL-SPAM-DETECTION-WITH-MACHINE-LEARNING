# EMAIL-SPAM-DETECTION-WITH-MACHINE-LEARNING
# Task  : EMAIL SPAM DETECTION WITH MACHINE LEARNING
### We’ve all been the recipient of spam emails before. Spam mail, or junk mail, is a type of email that is sent to a massive number of users at one time, frequently containing cryptic messages, scams, or most dangerously, phishing content.
### In this Project, use Python to build an email spam detector. Then, use machine learning to train the spam detector to recognize and classify emails into spam and non-spam. Let’s get started!

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer

### Load the dataset

data = pd.read_csv("spam.csv", encoding='latin-1')

data

### Data cleaning and preprocessing

data = data[['v1','v2']]

data

data.info()

data.shape

data['v1'].value_counts()

data['v1'] = data['v1'].map({"spam":0,"ham":1})

data.head()

y = data['v1']
x = data['v2']

x

y

### Splitting the dataset into testing and training dataset

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)

x_train

x_test

y_train.value_counts()

y_test.value_counts()

### converting word into vector by Tfidfvectorizer

vectorizer  = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print(x_train)

print(x_test)

### Train the Dataset by using Logistic Regression model

model = LogisticRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)

pred

### Model Evaluation

report = classification_report(y_test,pred)
print(report)

cm = confusion_matrix(y_test,pred)
cm

display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['ham','spam'])
display.plot()
plt.title("Confusion Matrix by Logistic Regression")

### Run the model

while True:
    x_new = input("Enter the Mail (or type 'exit' to exit): ")
    if x_new.lower() == 'exit':
        print("Exiting...")
        break
    input_data_feature = vectorizer.transform([x_new])
    prediction = model.predict(input_data_feature)
    if prediction[0] == 1:
        print("It is a ham mail.")
    else:
        print('It is a spam mail')

## The End
