import pandas as pd
import typing
from sklearn.metrics.cluster import normalized_mutual_info_score
from numpy import digitize, histogram_bin_edges
import matplotlib.pyplot as plt 
import numpy as np 
import ppscore as pps



@pd.api.extensions.register_dataframe_accessor('feature_importance')
class FeatureImportanceAccessor:

    def __init__(self, df):
        self._df = df
    
    def get_mi_score(
        x,
        y,
        digitization_bins=100
    ):
        ## da notare due cose: 
        # 1- la digitizzazione è necessaria per le variabili numeriche continue, per le variabili numeriche discrete è ininfluente, cambia solo le etichette
        # 2- normalized_mutual_info_score fornisce una misura di mutua informazione simmetrica
        x_dig=digitize(
            x=x,
            bins=histogram_bin_edges(
                a=x,
                bins=digitization_bins
            )
        )
        y_dig=digitize(
            x=y,
            bins=histogram_bin_edges(
                a=y,
                bins=digitization_bins
            )
        )
        return normalized_mutual_info_score(x_dig,y_dig)


    def feature_importance(
        self,
        target:str,
        score_type:str = 'MI'
    ):
        y = self._df[target]

        
        if score_type=='MI':
            scores = self._df.apply(
                lambda x: self.__get_mi_score(
                    x=x,
                    y=y
                ),
                axis=1
            )
        else:
            raise NotImplementedError("score types other than 'MI' aren't supported yet")








    def cross_corr(y, x, max_lag = 24, measure = 'Pearson', return_callable = False):
        l = y.shape[0]
        if measure == 'Pearson':
    #     x = (np.array(x) - np.mean(x))/np.std(x)
    #     y = (np.array(y) - np.mean(y))/np.std(y)
            cc = [np.corrcoef(y[lag:],x[:(l-lag)])[1,0] for lag in range(min(max_lag+1, l))]
        elif measure == 'PPS':
            cc = []
            for lag in range(min(max_lag+1, l)):
                df = pd.DataFrame()
                df['x'] = x[:(l-lag)]
                df['y'] = y[lag:]
                cc.append(
                    pps.score(df,'x','y')['ppscore']
                )
        if return_callable:
            return (lambda lag: cc[lag])
        return cc






    def plot_cross_corr(cc, title = None, **kwargs):
        fig = plt.figure(figsize = (10,3))
        plt.plot(cc,'ob', **kwargs)
        plt.title(title)
        plt.vlines(range(len(cc)), [0], cc)
        plt.show()



    def plot_cross_corr2(cc_train, cc_test, title_train = None, title_test = None, vline = None, **kwargs):
        fig,axs = plt.subplots(1,2,figsize = (20,3))
        axs[0].plot(cc_train,'ob', **kwargs)
        axs[0].set_title(title_train)
        axs[0].vlines(range(len(cc_train)), [0], cc_train)
        axs[1].plot(cc_test,'ob', **kwargs)
        axs[1].set_title(title_test)
        axs[1].vlines(range(len(cc_test)), [0], cc_test)
        if vline is not None:
            axs[0].axvline(x = vline, c = 'red')
            axs[1].axvline(x = vline, c = 'red')
        plt.show()



    def plot_cross_corr3(ccs, titles = None, vline = None, **kwargs):
        #n_plots_x = np.ceil(len(ccs)/2)
        fig,axs = plt.subplots(7, 2, figsize = (20,21))
        for ax,cc,title in zip(axs.flatten(),ccs,titles):
            ax.plot(cc,'ob', **kwargs)
            ax.vlines(range(len(cc)), [0], cc)
            ax.set_title(title)
            if vline is not None:
                ax.axvline(x = vline, c = 'red')
        plt.show()





    def filtered_cross_corr(target, feature, max_lag = 24, filter_arr = None, return_callable = False):
                        
        l = target.shape[0]
        
        ### se non viene passato nessun filtro sui dati metto a True tutti i valori di filter_arr
        if filter_arr is None:
            filter_arr = np.array([True]*l)
        
        lag_filter_arr = np.array([True]*l)  ### filtro booleano per escludere le osservazioni fino al lag di interesse
    #     print('lag_filter_arr_shape: ', lag_filter_arr.shape)
    #     print('filter_arr_shape: ', filter_arr.shape)
    #     print('target_shape: ', target.shape)
    #     print('feature_shape: ', feature.shape)
    #     print('feature_shape: ', feature.shape)
    #     print('mix_shape: ', (filter_arr & lag_filter_arr).shape)
        
        cc = []
        for lag in range(min(max_lag+1, l)):
            ### metto tutti i valori iniziali fino al lag voluto pari a False in modo che non vengano inclusi 
            lag_filter_arr[:lag] = False 

            
            cc.append( 
                np.corrcoef(
                    target[(filter_arr & lag_filter_arr)],   
                    feature.shift(lag)[(filter_arr & lag_filter_arr)] ### prima shifto la feature del lag voluto, poi filtro tramite
                                                            ###l'array di booleani filter_arr 
                )[1,0]  #corrcoef restituisce un matrice di correlazione, prendiamo solo il valore che ci interessa
            )
                
                
            
        
        if return_callable:
            return (lambda lag: cc[lag])
        return cc



