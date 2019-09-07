import pandas as pd
import datetime
import numpy as np
import math
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# Method for train and evaluate classifiers
def benchmark(clf, X_train, X_test, y_train, y_test, X_lately):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    predict = clf.predict(X_lately)
    print('The result with {} is {}'.format(clf.__class__.__name__, score))
    return clf.__class__.__name__, score, predict


def run():

    #Two options for read
    # First, download CSV data on https://finance.yahoo.com/quote/NFLX/history?p=NFLX
    df = pd.read_csv('NFLX.csv')
    print(df.head(10))

    #Second via web DataReader
    start = datetime.datetime(2014, 1, 2)
    end = datetime.datetime(2019, 9, 4)
    dfweb = web.DataReader("NFLX", 'yahoo', start, end)
    print(dfweb.head(10))

    df = dfweb

    #Rolling mean / Moving Average(MA)
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=10).mean()

    # Adjusting the size of matplotlib
    import matplotlib as mpl
    mpl.rc('figure', figsize=(8, 7))
    mpl.__version__

    # Adjusting the style of matplotlib
    style.use('ggplot')

    close_px.plot(label='NFLX')
    mavg.plot(label='mavg')
    plt.legend()
    plt.savefig('mean.png')
    plt.show()

    dfreg = df.loc[:, ['Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] *100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] *100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)

    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.05 * len(dfreg)))

    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)

    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifiers = []

    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    classifiers.append(clfreg)

    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)
    classifiers.append(clfknn)

    # SVR
    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    classifiers.append(svr_rbf)

    results = []
    for clf in classifiers:
        results.append(benchmark(clf,X_train, X_test, y_train, y_test, X_lately))

    dfreg['Forecast'] = np.nan
    dfreg.head(10)

    clf_scores = []
    clf_names = []

    last_date = dfreg.iloc[-1].name

    #Show result for each classifier
    for clf_name, score, forecast_set in results:
        clf_names.append(clf_name)
        clf_scores.append(score)

        print(last_date)
        last_unix = last_date
        next_unix = last_unix + datetime.timedelta(days=1)

        fig = plt.figure(len(clf_names))
        for i in forecast_set:
            next_date = next_unix
            next_unix += datetime.timedelta(days=1)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
        dfreg['Adj Close'].tail(1000).plot()
        dfreg['Forecast'].tail(1000).plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(clf_name + ' Predict price')
        plt.savefig(clf_name + '_predict.png')
        plt.show()

    fig = plt.figure(len(clf_name) + 1)
    bar_list = plt.bar(clf_names, clf_scores, color='blue')
    max_value = max(clf_scores)
    max_index = clf_scores.index(max_value)
    bar_list[max_index].set_color('g')
    plt.annotate(str(max_value)[:5] + ' - Best Model', xy=(clf_names[max_index], max_value), xytext=(clf_names[max_index], max_value*1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.xticks(clf_names)
    plt.xlabel('Classifier')
    plt.ylabel('Confidence')
    plt.title('Classifier Scores')
    plt.savefig('classifier_scores.png')
    plt.show()

if __name__ == '__main__':
    run()