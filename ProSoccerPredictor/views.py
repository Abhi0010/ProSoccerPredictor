from django.shortcuts import render, redirect
from django.forms import inlineformset_factory
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
import warnings
from .models import *
from .forms import CreateUserForm

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout

from django.contrib import messages

from django.contrib.auth.decorators import login_required


def home(request):
    return render(request, 'ProSoccerPredictor/home.html')


def about(request):
    return render(request, 'ProSoccerPredictor/about.html')


def contact(request):
    return render(request, 'ProSoccerPredictor/contact.html')


def register(request):
    if request.user.is_authenticated:
        return redirect('soccer-home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)

                return redirect('soccer-login')

        context = {'form': form}
    return render(request, 'ProSoccerPredictor/register.html', context)


def loginUser(request):
    if request.user.is_authenticated:
        return redirect('soccer-home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('soccer-home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'ProSoccerPredictor/login.html', context)


def logoutUser(request):
    logout(request)
    return redirect('soccer-login')


@login_required(login_url='soccer-login')
def myprofile(request):
    return render(request, 'ProSoccerPredictor/myprofile.html')


@login_required(login_url='soccer-login')
def predictor(request):
    return render(request, 'ProSoccerPredictor/predictor.html')


@login_required(login_url='soccer-login')
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
    dic = [{'win': outcome, 'home_goals': result1,
            'away_goals': result2, 'hometeam': hometeam, 'awayteam': awayteam}]
    # dic=df2.to_dict('records')

    context = {
        'post': dic
    }
    return render(request, 'ProSoccerPredictor/prediction.html', context)


@login_required(login_url='soccer-login')
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
    ax = sns.countplot(data['Preferred Foot'], palette='Greens')
    ax.set_title(
        label='Analysing players on the basis of preferred foot', fontsize=30)

    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri1 = urllib.parse.quote(string)

    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

# 2. Analysis based on different player positions

    plt.figure(figsize=(20, 8))
    ax = sns.countplot('Position', data=data, palette='bone')
    ax.set_xlabel(xlabel='Different Positions in Football', fontsize=16)
    ax.set_ylabel(ylabel='Player Count', fontsize=16)
    ax.set_title(
        label='Analysis based on different player positions', fontsize=30)

#    convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri2 = urllib.parse.quote(string)

# 3. Analysis based on Work rate of the players

    plt.figure(figsize=(15, 7))
    ax = sns.countplot(x='Work Rate', data=data, palette='husl')
    plt.title('Analysis based on Work rate of the players', fontsize=20)
    plt.xlabel('Work rates associated with the players', fontsize=20)
    plt.ylabel('Count of Players', fontsize=16)

    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri3 = urllib.parse.quote(string)

# 4 . Analysis based on skill moves of Players

    plt.figure(figsize=(10, 8))
    ax = sns.countplot(x='Skill Moves', data=data, palette='bright')
    ax.set_title(
        label='Analysis of players on basis of their skill moves', fontsize=20)
    ax.set_xlabel(xlabel='Number of Skill Moves', fontsize=16)
    ax.set_ylabel(ylabel='Count', fontsize=16)
    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri4 = urllib.parse.quote(string)

# 5 . # defining a function for cleaning the Weight data

    def extract_value_from(value):
        out = value.replace('lbs', '')
        return float(out)
    # applying the function to weight column
    data['Weight'] = data['Weight'].apply(lambda x: extract_value_from(x))

# Analysis based on body weight of the players

    plt.figure(figsize=(20, 5))
    plt.title('Analysis based on body weight of the players', fontsize=20)
    plt.xlabel('Weights associated with the players', fontsize=20)
    plt.ylabel('count of Players', fontsize=16)
    sns.set_style("darkgrid")
    ax = sns.distplot(data['Weight'], color='Black')

    # # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri5 = urllib.parse.quote(string)

    # return render(request, 'ProSoccerPredictor/analysis.html', {'data': uri})

    # Analysis based on potential scores of the players

    x = data.Potential
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-paper')
    sns.set_style("darkgrid")
    ax = sns.distplot(x, bins=58, kde=False, color='green')
    ax.set_xlabel(xlabel="Potential Scores of players", fontsize=16)
    ax.set_ylabel(ylabel='Number of players', fontsize=16)
    ax.set_title(
        label='Histogram for Potential Scores of Players', fontsize=20)

    # convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    ax.figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri6 = urllib.parse.quote(string)

    # 7 Analysis of overall scores of the players

    # sns.set(style="dark", palette="deep", color_codes=True)
    # x = data.Overall
    # plt.figure(figsize=(12, 8))
    # plt.style.use('ggplot')

    # ax = sns.distplot(x, bins=52, kde=False, color='b')
    # ax.set_xlabel(xlabel="Scores of players", fontsize=16)
    # ax.set_ylabel(ylabel='Number of players', fontsize=16)
    # ax.set_title(label='Histogram of players Overall Scores', fontsize=20)

    # # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri7 = urllib.parse.quote(string)

    # Analysis of different nations

    # plt.style.use('dark_background')
    # plt.title('Analysis of Different Nations Participating',
    #           fontsize=30, fontweight=20)
    # plt.xlabel('Name of The Country')
    # plt.ylabel('count')

    # ax = data['Nationality'].value_counts().head(
    #     80).plot.bar(color='red', figsize=(100, 70))

    # # convert graph into dtring buffer and then we convert 64 bit code into image
    # buf = io.BytesIO()
    # ax.figure.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri8 = urllib.parse.quote(string)
    plt.close('all')
    return render(request, 'ProSoccerPredictor/analysis.html', {'data1': uri1, 'data2': uri2, 'data3': uri3, 'data4': uri4, 'data5': uri5, 'data6': uri6})
