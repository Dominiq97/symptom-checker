from django.shortcuts import render, redirect
from django.views import generic
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
import regex as re
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from .forms import SymptomFormset
from .models import Symptom

val=None #valoare globala pt a lua variabile dintr o functie in alta

def create_diagnosis(request):
    global val
    template_name = 'store/create_diagnosis.html'
    heading_message = 'Introduce your symptoms here'
    if request.method == 'GET':
        formset = SymptomFormset(request.GET or None)
    elif request.method == 'POST':
        formset = SymptomFormset(request.POST)
        if formset.is_valid():
            for form in formset:
                name = form.cleaned_data.get('name')
                if name:
                    Symptom(name=name).save()
            str = '';
            for i in formset.cleaned_data:
                str = str+' '+i['name']
            def val():
                return str
            return redirect('store:result')

    return render(request, template_name, {
        'formset': formset,
        'heading': heading_message,
    })


def result(request):
    df = pd.read_csv('C:\\Users\\gigel\\Desktop\\facultate\\Software Engineering\\symptom-checker\\sc_scripts\\dataset_prelucrat.csv',engine='python')
    df = df.fillna(0)

    fill = df['Disease'].iloc[0]
    for i in range(1,1866):
        if df['Disease'].iloc[i] == 0:
            df['Disease'].iloc[i] = fill
        else:
            fill = df['Disease'].iloc[i]
    df['Disease']

    fill = df['Count of Disease Occurrence'].iloc[0]
    # print(fill)
    for i in range(1,1866):
        if df['Count of Disease Occurrence'].iloc[i] == 0.0:
            df['Count of Disease Occurrence'].iloc[i] = fill
        else:
            fill = df['Count of Disease Occurrence'].iloc[i]

    df = df[df.Symptom != 0]

    df['Symptom'] = df['Symptom'].apply(lambda x: x.split('^'))
    df['Symptom']

    df = df.explode('Symptom').reset_index()

    df.Symptom = df.Symptom.apply(lambda x: x.split('_')[0])

    df['Disease'] = df['Disease'].apply(lambda x: x.split('^'))
    df = df.explode('Disease').reset_index()
    df.Disease = df.Disease.apply(lambda x: x.split('_')[0])

    df.drop(['index', 'level_0','Count of Disease Occurrence'], axis = 1, inplace = True)
    df_sparse = pd.get_dummies(df, columns = ['Symptom']).drop_duplicates()

    df_sparse = df_sparse.groupby('Disease').sum().reset_index()

    X = df_sparse[df_sparse.columns[1:]]
    Y = df_sparse['Disease']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    input_data = pd.read_csv('C://Users//gigel//Desktop//facultate//Training.csv',engine='python')
    # print(input_data.head())
    test_data = pd.read_csv('C://Users//gigel//Desktop//facultate//Testing.csv',engine='python')
    # print(test_data.head())

    # print(input_data.shape)
    input_data.isnull().sum().sort_values(ascending=False)
    input_data['prognosis'].value_counts(normalize = True)

    from scipy.stats import chi2_contingency
    chi2_contingency(pd.crosstab(input_data['cold_hands_and_feets'],input_data['weight_gain']))

    x = input_data.drop(['prognosis'],axis =1)
    y = input_data['prognosis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    from sklearn.naive_bayes import MultinomialNB

    mnb = MultinomialNB()
    mnb = mnb.fit(x_train, y_train)
    score = mnb.score(x_test, y_test)

    gbm_clf = GradientBoostingClassifier()
    gbm_clf.fit(x_train, y_train)
    score = gbm_clf.score(x_train, y_train)

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

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
    importances = gbm.feature_importances_
    indices = np.argsort(importances)[::-1]

    features = input_data.columns[:-1]

    feature_dict = {}
    for i,f in enumerate(features):
        feature_dict[f] = i

    sample_x = [i/52 if i ==52 else i/24 if i==24 else i*0 for i in range(len(features))]

    sample_x = np.array(sample_x).reshape(1,len(sample_x))

    symptoms = x.columns

    regex = re.compile('_')

    symptoms = [i if regex.search(i) == None else i.replace('_', ' ') for i in symptoms ]

    from difflib import get_close_matches
    def closeMatches(patterns, word):
        print(get_close_matches(word, patterns, n=2, cutoff=0.7))

    from flashtext import KeywordProcessor
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(symptoms)

    def predict_disease(query):
        matched_keyword = keyword_processor.extract_keywords(query)
        if len(matched_keyword) == 0:
            return "No Matches"
        else:
            regex = re.compile(' ')
            processed_keywords = [i if regex.search(i) == None else i.replace(' ', '_') for i in matched_keyword]
            coded_features = []
            for keyword in processed_keywords:
                coded_features.append(feature_dict[keyword])
            sample_x = []
            for i in range(len(features)):
                try:
                    sample_x.append(i/coded_features[coded_features.index(i)])
                except:
                    sample_x.append(i*0)
            sample_x = np.array(sample_x).reshape(1,len(sample_x))
            return gbm.predict(sample_x)[0]


    # query = 'I have anxiety and fatigue'
    res = val() # ia resultatu cu val() si l duce in result.html
    result = predict_disease(res)

    template_name='store/result.html'
    return render(request, template_name, {'result':result})


class SymptomListView(generic.ListView):
    model = Symptom
    context_object_name = 'symptoms'
    template_name = 'store/list.html'



