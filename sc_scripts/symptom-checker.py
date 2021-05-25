import pandas as pd
import numpy as np
import os
import regex as re
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('C://Users//gigel//Desktop//facultate//dataset_prelucrat.csv',engine='python')
print(df.count())



print(df.head())
df = df.fillna(0)

fill = df['Disease'].iloc[0]
for i in range(1,1866):
    if df['Disease'].iloc[i] == 0:
        df['Disease'].iloc[i] = fill
    else:
        fill = df['Disease'].iloc[i]
df['Disease']

fill = df['Count of Disease Occurrence'].iloc[0]
print(fill)
for i in range(1,1866):
    if df['Count of Disease Occurrence'].iloc[i] == 0.0:
        df['Count of Disease Occurrence'].iloc[i] = fill
    else:
        fill = df['Count of Disease Occurrence'].iloc[i]

print(df.head())

df = df[df.Symptom != 0]

df['Symptom'] = df['Symptom'].apply(lambda x: x.split('^'))
df['Symptom']

df = df.explode('Symptom').reset_index()

df.Symptom = df.Symptom.apply(lambda x: x.split('_')[0])
print(df)

df['Disease'] = df['Disease'].apply(lambda x: x.split('^'))
df = df.explode('Disease').reset_index()
df.Disease = df.Disease.apply(lambda x: x.split('_')[0])
print(df)

df.drop(['index', 'level_0','Count of Disease Occurrence'], axis = 1, inplace = True)
print(df)

df_sparse = pd.get_dummies(df, columns = ['Symptom']).drop_duplicates()
print(df_sparse.head())

df_sparse = df_sparse.groupby('Disease').sum().reset_index()
print(df_sparse.head())

X = df_sparse[df_sparse.columns[1:]]
Y = df_sparse['Disease']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
# xgb_clf = GradientBoostingClassifier()
# xgb_clf.fit(X, Y)
# score = xgb_clf.score(X, Y)
# print(score)

# print ("DecisionTree")
# clf = DecisionTreeClassifier()
# model = clf.fit(X,Y)
# print ("Acurracy: ", model.score(X,Y))

# print(model.predict(x_test))
# print(model.score(x_test, y_test))



input_data = pd.read_csv('C://Users//gigel//Desktop//facultate//Training.csv',engine='python')
print(input_data.head())
test_data = pd.read_csv('C://Users//gigel//Desktop//facultate//Testing.csv',engine='python')
print(test_data.head())

print(input_data.shape)
input_data.isnull().sum().sort_values(ascending=False)
input_data['prognosis'].value_counts(normalize = True)

# input_data['prognosis'].value_counts(normalize = True).plot.bar(color='red')
# plt.subplots_adjust(left = 0.9, right = 2 , top = 2, bottom = 1)
# plt.show()

# corr = input_data.corr()
# mask = np.array(corr)
# mask[np.tril_indices_from(mask)] = False
# plt.subplots_adjust(left = 0.5, right = 16 , top = 20, bottom = 0.5)
# sns.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True,cmap="YlGnBu")
# plt.show()

print(pd.crosstab(input_data['cold_hands_and_feets'],input_data['weight_gain']))

from scipy.stats import chi2_contingency
#as p value is  0.0  which is less than 0.05 then they are actually different from each other which satisfy the alternate hypothesis 
chi2_contingency(pd.crosstab(input_data['cold_hands_and_feets'],input_data['weight_gain']))
print(chi2_contingency(pd.crosstab(input_data['cold_hands_and_feets'],input_data['weight_gain'])))

x = input_data.drop(['prognosis'],axis =1)
y = input_data['prognosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB

#fitted the model
mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)

score = mnb.score(x_test, y_test)
print("Accuracy Score: ",score)

gbm_clf = GradientBoostingClassifier()
gbm_clf.fit(x_train, y_train)
score = gbm_clf.score(x_train, y_train)
print(score)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())

scores = cross_val_score(gbm_clf, x_test, y_test, cv=10)
print (scores)
print (scores.mean())

real_diseases = y_test.values
y_pred = gbm_clf.predict(x_test)
#for the cross checking purpose i want to see if predicted values and actual values are same else it gives me worng prediction 
for i in range(0, 20):
    if y_pred[i] == real_diseases[i]:
        print ('Pred: {0} Actual:{1}'.format(y_pred[i], real_diseases[i]))
    else:
        print('worng prediction')
        print ('Pred: {0} Actual:{1}'.format(y_pred[i], real_diseases[i]))

#imported Kfold
from sklearn.model_selection import KFold

## Function to run multiple algorithms with different K values of KFold.
def evaluate(train_data,kmax,algo):
    test_scores = {}
    train_scores = {}
    for i in range(2,kmax,2):
        kf = KFold(n_splits = i)
        sum_train = 0
        sum_test = 0
        data = input_data
        for train,test in kf.split(data):
            train_data = data.iloc[train,:]
            test_data = data.iloc[test,:]
            x_train = train_data.drop(["prognosis"],axis=1)
            y_train = train_data['prognosis']
            x_test = test_data.drop(["prognosis"],axis=1)
            y_test = test_data["prognosis"]
            algo_model = algo.fit(x_train,y_train)
            sum_train += algo_model.score(x_train,y_train)
            y_pred = algo_model.predict(x_test)
            sum_test += accuracy_score(y_test,y_pred)
        average_test = sum_test/i
        average_train = sum_train/i
        test_scores[i] = average_test
        train_scores[i] = average_train
        print("kvalue: ",i)
    return(train_scores,test_scores)

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
nb = MultinomialNB()
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',)
from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier(n_estimators = 10)

algo_dict = {'l_o_g':log,'d_t':dt,'r_a_n':ran,'N_B' : nb, 'G_B' : gbm}
algo_train_scores={}
algo_test_scores={}

# max_kfold = 11
# for algo_name in algo_dict.keys():
#     print(algo_name)
#     tr_score,tst_score = evaluate(input_data,max_kfold,algo_dict[algo_name])
#     algo_train_scores[algo_name] = tr_score
#     algo_test_scores[algo_name] = tst_score
# print(algo_train_scores)
# print(algo_test_scores)

# df_test = pd.DataFrame(algo_test_scores)
# df_train = pd.DataFrame(algo_train_scores)

# df_test.plot(grid = 1)
# plt.show()

#building the model at k value 2 
test_scores={}
train_scores={}
for i in range(2,4,2):
    kf = KFold(n_splits = i)
    sum_train = 0
    sum_test = 0
    data = input_data
    for train,test in kf.split(data):
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        x_train = train_data.drop(["prognosis"],axis=1)
        y_train = train_data['prognosis']
        x_test = test_data.drop(["prognosis"],axis=1)
        y_test = test_data["prognosis"]
        algo_model = gbm.fit(x_train,y_train)
        sum_train += gbm.score(x_train,y_train)
        y_pred = gbm.predict(x_test)
        sum_test += accuracy_score(y_test,y_pred)
    average_test = sum_test/i
    average_train = sum_train/i
    test_scores[i] = average_test
    train_scores[i] = average_train
    print("kvalue: ",i)

print(train_scores)
print(test_scores)

importances = gbm.feature_importances_
indices = np.argsort(importances)[::-1]

features = input_data.columns[:-1]
for f in range(5):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i

print(feature_dict['redness_of_eyes'], feature_dict['cough'])



sample_x = [i/52 if i ==52 else i/24 if i==24 else i*0 for i in range(len(features))]
len(sample_x)

sample_x = np.array(sample_x).reshape(1,len(sample_x))
print(gbm.predict(sample_x))

print(gbm.predict_proba(sample_x))
print(gbm.__getstate__())

symptoms = x.columns

regex = re.compile('_')

symptoms = [i if regex.search(i) == None else i.replace('_', ' ') for i in symptoms ]

# Function to find all close matches of  
# input string in given list of possible strings 
from difflib import get_close_matches  
def closeMatches(patterns, word): 
    print(get_close_matches(word, patterns, n=2, cutoff=0.7))

word = 'sivering'
closeMatches(symptoms, word)

from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(symptoms)

text = 'I have itching, joint pain and fatigue'
keyword_processor.extract_keywords(text)

def predict_disease(query):
    matched_keyword = keyword_processor.extract_keywords(query)
    if len(matched_keyword) == 0:
        print("No Matches")
    else:
        regex = re.compile(' ')
        processed_keywords = [i if regex.search(i) == None else i.replace(' ', '_') for i in matched_keyword]
        print(processed_keywords)
        coded_features = []
        for keyword in processed_keywords:
            coded_features.append(feature_dict[keyword])
        #print(coded_features)
        sample_x = []
        for i in range(len(features)):
            try:
                sample_x.append(i/coded_features[coded_features.index(i)])
            except:
                sample_x.append(i*0)
        sample_x = np.array(sample_x).reshape(1,len(sample_x))
        print('Predicted Disease: ',gbm.predict(sample_x)[0])
                

query = 'I have redness of eyes and cough'

predict_disease(query)






