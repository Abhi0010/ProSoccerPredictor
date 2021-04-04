from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
import sklearn
import joblib
from django.http import HttpResponse
from matplotlib import pylab
# from pylab import *
# import PIL, PIL.Image, StringIO
import matplotlib.pyplot as plt
import io
import urllib
import base64


def home(request):
    return render(request, 'ProSoccerPredictor/home.html')


def about(request):
    return render(request, 'ProSoccerPredictor/about.html')


def contact(request):
    return render(request, 'ProSoccerPredictor/contact.html')


def login(request):
    return render(request, 'ProSoccerPredictor/login.html')


def register(request):
    return render(request, 'ProSoccerPredictor/register.html')


def myprofile(request):
    return render(request, 'ProSoccerPredictor/myprofile.html')


def predictor(request):
    return render(request, 'ProSoccerPredictor/predictor.html')


def prediction(request):
    df = pd.read_csv("dataset_preprocess.csv")
    dic = []

    homecol = df['HomeTeam'].values
    awaycol = df['AwayTeam'].values
    # hometeam = request.POST.get('home')
    # awayteam = request.POST.get('away')
    # # hometeam = request.POST['home']
    # # awayteam = request.POST['away']
    dic_result = {}
    dic_result = request.POST
    hometeam = dic_result['home']
    awayteam = dic_result['away']
    homelastloc = max(loc for loc, val in enumerate(
        homecol) if val == hometeam)
    awaylastloc = max(loc for loc, val in enumerate(
        awaycol) if val == awayteam)

    df_home = df.iloc[homelastloc]
    df_away = df.iloc[awaylastloc]

    df_home = df_home.drop(labels=['HomeTeam', 'AwayTeam', 'id'])
    df_away = df_away.drop(labels=['HomeTeam', 'AwayTeam', 'id'])

    df_list = []
    df_list.append(df_home['HTP'])
    df_list.append(df_away['ATP'])
    for i in range(3, 15):
        df_list.append(df_home[i])
    for i in range(15, 27):
        df_list.append(df_away[i])
    df_list.append(df_home['HTGD'])
    df_list.append(df_away['ATGD'])
    # df2=pd.DataFrame(df2.drop(labels=['HomeTeam', 'AwayTeam','id']))
    # df2=df2.T

    # arr = np.array(df2_list)
    # inp = np.reshape(arr, (-1, -1))
    # inp=list(inp)
    cls = joblib.load('svm.sav')
    result = cls.predict([df_list])

    cls = joblib.load('home_goals.sav')
    result1 = cls.predict([df_list])

    cls = joblib.load('away_goals.sav')
    result2 = cls.predict([df_list])
    result1 = int(np.round(result1))
    result2 = int(np.round(result2))

    if(result == 0):
        outcome = hometeam
    elif(result == 1):
        outcome = awayteam
    else:
        outcome = 'Draw'
    dic = [{'win': outcome, 'home_goals': result1, 'away_goals': result2}]
    # dic=df2.to_dict('records')

    context = {
        'post': dic
    }

    return render(request, 'ProSoccerPredictor/prediction.html', context)


def analysis(request):
    data = pd.read_csv("data.csv")

    data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace=True)
    data['Volleys'].fillna(data['Volleys'].mean(), inplace=True)
    data['Dribbling'].fillna(data['Dribbling'].mean(), inplace=True)
    data['Curve'].fillna(data['Curve'].mean(), inplace=True)
    data['FKAccuracy'].fillna(data['FKAccuracy'], inplace=True)
    data['LongPassing'].fillna(data['LongPassing'].mean(), inplace=True)
    data['BallControl'].fillna(data['BallControl'].mean(), inplace=True)
    data['HeadingAccuracy'].fillna(
        data['HeadingAccuracy'].mean(), inplace=True)
    data['Finishing'].fillna(data['Finishing'].mean(), inplace=True)
    data['Crossing'].fillna(data['Crossing'].mean(), inplace=True)
    data['Weight'].fillna('200lbs', inplace=True)
    data['Contract Valid Until'].fillna(2019, inplace=True)
    data['Height'].fillna("5'11", inplace=True)
    data['Loaned From'].fillna('None', inplace=True)
    data['Joined'].fillna('Jul 1, 2018', inplace=True)
    data['Jersey Number'].fillna(8, inplace=True)
    data['Body Type'].fillna('Normal', inplace=True)
    data['Position'].fillna('ST', inplace=True)
    data['Club'].fillna('No Club', inplace=True)
    data['Work Rate'].fillna('Medium/ Medium', inplace=True)
    data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace=True)
    data['Weak Foot'].fillna(3, inplace=True)
    data['Preferred Foot'].fillna('Right', inplace=True)
    data['International Reputation'].fillna(1, inplace=True)
    data['Wage'].fillna('€200K', inplace=True)
    data.fillna(0, inplace=True)

# 1. Analysing players on the basis of preferred foot (Right or Left)
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.title('Analysis on the basis of preferred foot of the players', fontsize=40)
    ax = sns.countplot(data['Preferred Foot'], palette='Greens')

    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

# 2. Analysis based on different player positions

    # plt.figure(figsize = (20, 8))
    # ax = sns.countplot('Position', data = data, palette = 'bone')
    # ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
    # ax.set_ylabel(ylabel = 'Player Count', fontsize = 16)
    # ax.set_title(label = 'Analysis based on different player positions', fontsize = 30)

   # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri = urllib.parse.quote(string)
    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

# 3. Analysing the players on the basis of Wages

    # Defining a function for cleaning the wage column

    # def extract_value_from(Value):
    #     out = Value.replace('€', '')
    #     if 'M' in out:
    #         out = float(out.replace('M', ''))*1000000
    #     elif 'K' in Value:
    #         out = float(out.replace('K', ''))*1000
    #     return float(out)

    # #Applying the function to the wage column

    # data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))
    # data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))

    # data['Wage'].head()

    # import warnings
    # warnings.filterwarnings('ignore')

    # plt.rcParams['figure.figsize'] = (15, 5)
    # ax = sns.distplot(data['Wage'], color = 'red')
    # plt.xlabel('Wage Range for Players', fontsize = 16)
    # plt.ylabel('Count of the Players', fontsize = 16)
    # plt.title('Analysis of Wages of Players', fontsize = 30)
    # plt.xticks(rotation = 90)

    # # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri = urllib.parse.quote(string)

    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

# 4 . Analysis based on skill moves of Players

    # plt.figure(figsize = (10, 8))
    # ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'bright')
    # ax.set_title(label = 'Analysis of players on basis of their skill moves', fontsize = 20)
    # ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)
    # ax.set_ylabel(ylabel = 'Count', fontsize = 16)
    # # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri = urllib.parse.quote(string)

    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

# 5 . Analysing players on basis of height

    # plt.figure(figsize=(15, 10))
    # ax = sns.countplot(x='Height', data=data, palette='muted')
    # ax.set_title(
    #     label='Analysis of players based on their height', fontsize=20)
    # ax.set_xlabel(xlabel='Height in Foot per inch', fontsize=16)
    # ax.set_ylabel(ylabel='Count', fontsize=16)
    # # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri = urllib.parse.quote(string)

    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

    # def extract_value_from(value):
    # out = value.replace('lbs', '')
    # return float(out)

# 6 .#Analysis based on body weight of the players
    # applying the function to weight column
    # defining a function for cleaning the Weight data

    # def extract_value_from(value):
    #     out = value.replace('lbs', '')
    #     return float(out)

    # data['Weight'] = data['Weight'].apply(lambda x: extract_value_from(x))

    # data['Weight'].head()

    # plt.figure(figsize=(20, 5))
    # sns.set_style("darkgrid")
    # ax = sns.distplot(data['Weight'], color='Black')
    # plt.title('Analysis based on body weight of the players', fontsize=20)
    # plt.xlabel('Weights associated with the players', fontsize=20)
    # plt.ylabel('count of Players', fontsize=16)

    # # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri = urllib.parse.quote(string)

    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

    # 7. Analysis based on Work rate of the players

    # plt.figure(figsize = (15, 7))
    # sns.countplot(x = 'Work Rate', data = data, palette = 'husl')
    # plt.title('Analysis based on Work rate of the players', fontsize = 20)
    # plt.xlabel('Work rates associated with the players', fontsize = 20)
    # plt.ylabel('Count of Players', fontsize = 16)
    # plt.show()
