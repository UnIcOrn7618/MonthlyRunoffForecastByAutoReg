# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
from scipy import linalg
import pandas as pd

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print(__doc__)

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'\\results_analysis\\graphs\\'
# #############################################################################
# Create the data



def model_select_pca(station,decomposer,predict_pattern,ax=None,wavelet_level='db10-2'):
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    # Set parameters for PCA
    # load one-step one-month forecast or hindcast samples and the normalization indicators
    if decomposer=='dwt':
        train = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/minmax_unsample_train.csv')
        dev = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/minmax_unsample_dev.csv')
        test = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/minmax_unsample_test.csv')
        norm_id = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/norm_unsample_id.csv')
    else:
        train = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/minmax_unsample_train.csv')
        dev = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/minmax_unsample_dev.csv')
        test = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/minmax_unsample_test.csv')
        norm_id = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/norm_unsample_id.csv')
    sMax = (norm_id['series_max']).values
    sMin = (norm_id['series_min']).values
    # Conncat the training, development and testing samples
    samples = pd.concat([train,dev,test],axis=0)
    samples = samples.reset_index(drop=True)
    # Renormalized the entire samples
    samples = np.multiply(samples + 1,sMax - sMin) / 2 + sMin

    y = samples['Y']
    X = samples.drop('Y',axis=1)
    
    # #############################################################################
    # Fit the models
    n_features = X.shape[1]
    print("Number of features:{}".format(n_features))
    n_components = np.arange(0, n_features, 5)  # options for n_components
    def compute_scores(X):
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis()

        pca_scores, fa_scores = [], []
        for n in n_components:
            pca.n_components = n
            fa.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
            fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

        return pca_scores, fa_scores
    
    
    def shrunk_cov_score(X):
        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=5)
        return np.mean(cross_val_score(cv.fit(X).best_estimator_, X, cv=5))


    def lw_score(X):
        return np.mean(cross_val_score(LedoitWolf(), X, cv=5))


    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]
    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_
    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    if ax==None:
        plt.figure(figsize=(5.5139,3.5139))
        plt.plot(n_components, pca_scores, 'b', label='PCA scores')
        plt.plot(n_components, fa_scores, 'r', label='FA scores')
        plt.axvline(n_components_pca, color='b',
                    label='PCA CV: %d' % n_components_pca, linestyle='--')
        plt.axvline(n_components_fa, color='r',
                    label='FactorAnalysis CV: %d' % n_components_fa,
                    linestyle='--')
        plt.axvline(n_components_pca_mle, color='k',
                    label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
        # compare with other covariance estimators
        plt.axhline(shrunk_cov_score(X), color='violet',
                    label='Shrunk Covariance MLE', linestyle='-.')
        plt.axhline(lw_score(X), color='orange',
                    label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')
        plt.xlabel('Number of components')
        plt.ylabel('CV scores')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(graphs_path+"two_stage_pca_fa_analysis_"+station+"_"+decomposer+".eps",format="EPS",dpi=2000)
        plt.savefig(graphs_path+"two_stage_pca_fa_analysis_"+station+"_"+decomposer+".tif",format="TIFF",dpi=1200)
        plt.show()
    else:
        ax.plot(n_components, pca_scores, 'b', label='PCA scores')
        ax.plot(n_components, fa_scores, 'r', label='FA scores')
        ax.axvline(n_components_pca, color='b',
                    label='PCA CV: %d' % n_components_pca, linestyle='--')
        ax.axvline(n_components_fa, color='r',
                    label='FactorAnalysis CV: %d' % n_components_fa,
                    linestyle='--')
        ax.axvline(n_components_pca_mle, color='k',
                    label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
        # compare with other covariance estimators
        ax.axhline(shrunk_cov_score(X), color='violet',
                    label='Shrunk Covariance MLE', linestyle='-.')
        ax.axhline(lw_score(X), color='orange',
                    label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('CV scores')
        ax.legend(loc='lower right')
        plt.tight_layout()
        # plt.savefig(graphs_path+"two_stage_pca_fa_analysis_"+station+"_"+decomposer+".eps",format="EPS",dpi=2000)
        # plt.savefig(graphs_path+"two_stage_pca_fa_analysis_"+station+"_"+decomposer+".tif",format="TIFF",dpi=1200)
        # plt.show()

    return ax

if __name__ == "__main__":
    plt.figure(figsize=(7.48,7.48))
    ax1=plt.subplot(3,4,1)
    ax2=plt.subplot(3,4,2)
    ax3=plt.subplot(3,4,3)
    ax4=plt.subplot(3,4,4)
    ax5=plt.subplot(3,4,5)
    ax6=plt.subplot(3,4,6)
    ax7=plt.subplot(3,4,7)
    ax8=plt.subplot(3,4,8)
    ax9=plt.subplot(3,4,9)
    ax10=plt.subplot(3,4,10)
    ax11=plt.subplot(3,4,11)
    ax12=plt.subplot(3,4,12)
    model_select_pca(station='Huaxian',decomposer='eemd',predict_pattern='one_step_1_month_forecast',ax=ax1)
    model_select_pca(station='Huaxian',decomposer='ssa',predict_pattern='one_step_1_month_forecast',ax=ax2)
    model_select_pca(station='Huaxian',decomposer='vmd',predict_pattern='one_step_1_month_forecast',ax=ax3)
    model_select_pca(station='Huaxian',decomposer='dwt',predict_pattern='one_step_1_month_forecast',ax=ax4)
    model_select_pca(station='Xianyang',decomposer='eemd',predict_pattern='one_step_1_month_forecast',ax=ax5)
    model_select_pca(station='Xianyang',decomposer='ssa',predict_pattern='one_step_1_month_forecast',ax=ax6)
    model_select_pca(station='Xianyang',decomposer='vmd',predict_pattern='one_step_1_month_forecast',ax=ax7)
    model_select_pca(station='Xianyang',decomposer='dwt',predict_pattern='one_step_1_month_forecast',ax=ax8)
    model_select_pca(station='Zhangjiashan',decomposer='eemd',predict_pattern='one_step_1_month_forecast',ax=ax9)
    model_select_pca(station='Zhangjiashan',decomposer='ssa',predict_pattern='one_step_1_month_forecast',ax=ax10)
    model_select_pca(station='Zhangjiashan',decomposer='vmd',predict_pattern='one_step_1_month_forecast',ax=ax11)
    model_select_pca(station='Zhangjiashan',decomposer='dwt',predict_pattern='one_step_1_month_forecast',ax=ax12)
    plt.savefig(graphs_path+"PCA_FA_analysis.eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"PCA_FA_analysis.tif",format="TIFF",dpi=1200)
    plt.show()