#!/usr/bin/python

from sklearn.svm import NuSVR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # scoring: Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring='r2',cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return fig

def tune_NuSVR(X_tr, y_tr):
    print 'tune_nuSVR(X_tr, y_tr):'

    regressor = NuSVR()
    # param_space = {'C': [0.1, 0.3, 0.5, 0.7, 1.0],
    #                'nu': [0.1, 0.3, 0.5, 0.7, 1.0], 
    #                'kernel': ['linear', 'rbf', 'poly'], 
    #                'degree': [3, 5, 7], 
    #                'gamma': [0.0, 0.1, 0.3, 0.5, 0.7]}

    # dummy
    param_space = {'C': [1.0],
                   'nu': [0.5], 
                   'kernel': ['rbf'], 
                   'degree': [3], 
                   'gamma': [0.7]}

    #
    kf_cv = cross_validation.KFold(n=len(y_tr), n_folds=10)
    grid_search = GridSearchCV(regressor, param_grid=param_space, cv=kf_cv)# use r2_score by default for regression
    grid_search.fit(X_tr,y_tr)# Run fit with all sets of parameters.

    return grid_search.best_estimator_

def train(estimator, X_tr, y_tr):
    estimator.fit(X_tr, y_tr)
    return estimator

def test(estimator, X_te, y_te):
    y_pred = estimator.predict(X_te)
    y_true = y_te

    perf = {}
    perf['mse'] = mean_squared_error(y_true, y_pred) 
    perf['r2'] = r2_score(y_true, y_pred)
    perf['y_pred'] = y_pred
    perf['y_true'] = y_true
    perf['estimator'] = estimator
     
    return perf

def get_best_perf(perfs, scoring):
    score_list = [perf[scoring] for perf in perfs]

    best_idx = None
    if scoring=='mse':
        best_idx = score_list.index( min(score_list) )
    elif scoring=='r2':
        best_idx = score_list.index( max(score_list) )
    else:
        assert False, 'UNKNOWN scoring :('

    return perfs[best_idx]

def main():
    #
    regression_data_dir = '/home/ian-djakman/Documents/data/output/knowledge-compatibility-benchmarker/regression-data'
    out_dir = '/home/ian-djakman/Documents/data/output/knowledge-compatibility-benchmarker/regression-eval'

    regression_data_filename  = 'voc2010_philippunary.csv'
    regression_data_filepath = regression_data_dir + '/' + regression_data_filename
    data = np.genfromtxt(regression_data_filepath, delimiter=',')
    n_sample = data.shape[0]

    #
    target_col = -1
    X = data[:,0:target_col]
    y = data[:,target_col]


    fig = plt.figure()
    plt.scatter(range(len(y)), y)
    plt.grid(True)
    plt.xlabel('$i^{th}$ sample')
    plt.ylabel('$y_{true}$')
    if out_dir!=None:
        with PdfPages(out_dir+'/target_scatter_plot.pdf') as pdf:
            pdf.savefig(fig)

    # Shuffle n_clone times
    # NOTE: a single dataset is a list of [X_tr, X_te, y_tr, y_te]
    n_clone = 1
    datasets = [train_test_split(X, y, test_size=0.3, random_state=i) for i in range(n_clone)]

    #
    perf_of_datasets = []
    regressors = []
    for dataset in datasets:
        X_tr, X_te, y_tr, y_te = dataset

        meta_regressor = tune_NuSVR(X_tr, y_tr)
        regressor = train(meta_regressor, X_tr, y_tr)
        regressors.append(regressor)

        #  
        print 'plot_learning_curve()'
        title = 'Learning curve'
        ylim = (0.0, 1.0)
        fig = plot_learning_curve(regressor, title, X_tr, y_tr, ylim, cv=10, n_jobs=4)
        if out_dir!=None:
            with PdfPages(out_dir+'/learning_curve.pdf') as pdf:
                pdf.savefig(fig)

        #
        perf = test(regressor, X_te, y_te)
        perf_of_datasets.append(perf)

    #
    best_perf_basedon = {}
    best_perf_basedon['mse'] = get_best_perf(perf_of_datasets, 'mse')
    best_perf_basedon['r2'] = get_best_perf(perf_of_datasets, 'r2')

    #
    for scoring, perf in best_perf_basedon.iteritems():
        #
        fig, ax = plt.subplots()
        scatter_plot = ax.scatter(perf['y_true'], perf['y_pred'])
        ax.plot([0.0, 1.0], [0.0, 1.0], '-', linewidth=2, color='red')

        ax.set_ylabel('$y_{pred}$')
        ax.set_xlabel('$y_{true}$')
        ax.set_title( 'Testing NuSVR: ' + scoring + ' =' + str(perf[scoring]))
        
        xlim = (-0.2, 1.2)
        ylim = xlim
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax.grid(True)

        with PdfPages(out_dir + '/best_ypred_vs_ytrue_based_on_'+scoring+'.pdf') as pdf:
            pdf.savefig(fig)

        #
        with open(out_dir+'/best_regressor_based_on_'+scoring+'.param', 'w') as f:
            f.write(str(perf['estimator']))

if __name__ == '__main__':
    main()
