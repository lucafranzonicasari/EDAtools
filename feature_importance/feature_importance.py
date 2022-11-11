import pandas as pd
import typing
from sklearn.metrics.cluster import normalized_mutual_info_score
from numpy import digitize, histogram_bin_edges
import matplotlib.pyplot as plt 
import numpy as np 
import ppscore as pps
from scipy.stats import pearsonr
from tqdm.notebook import tqdm



@pd.api.extensions.register_dataframe_accessor('feature_importance')
class FeatureImportanceAccessor:

    def __init__(self, df):
        self._df = df
    
    def get_score(self,
        x,
        y,
        target_type:str,
        score_type:str,
        **kwargs
    ):

        ### validation
        if type(x) is str:
            x=self._df[x]
        else:
            raise Exception("'x' must be of type str (name of a column of the pandas.DataFrame instance the method is called on)")


        if type(y) is str:
            y=self._df[y]
        elif type(y) is pd.Series:
            if not x.index.equals(y.index):
                raise Exception("when 'y' is passed as an external pandas.Series instance, it must have the same index as the pandas.DataFrame instance the method is called on")
        else:
            raise Exception("'y' must be either of type str (name of a column of the dataframe the method is called on) or of type pandas.Series (external target)")   

        if target_type not in ['reg', 'clas']:
            raise Exception("'target_type' must be either 'reg' or 'clas'")




        if score_type == 'mi':
            if 'bins' not in kwargs:
                kwargs['bins'] = 50
            ## da notare due cose: 
            # 1- la digitizzazione è necessaria per le variabili numeriche continue, per le variabili numeriche discrete è ininfluente, cambia solo le etichette
            # 2- normalized_mutual_info_score fornisce una misura di mutua informazione simmetrica
            x_dig=digitize(
                x=x,
                bins=histogram_bin_edges(
                    a=x,
                    **kwargs
                )
            )
            y_dig=digitize(
                x=y,
                bins=histogram_bin_edges(
                    a=y,
                    **kwargs
                )
            )
            score = normalized_mutual_info_score(
                x_dig,
                y_dig
            )
        elif score_type == 'pp':
            if 'random_seed' not in kwargs:
                kwargs['random_seed'] = 0
            # ppscore functions determins whether to use a classification or a regression model based on the data type of the target
            score = pps.score(self._df, 
                x, 
                y if target_type=='reg' else y.astype('category'), 
                **kwargs
            )
        
        elif score_type == 'r':
            score = pearsonr( 
                x, 
                y
            )[0]
        else:
            raise Exception("allowed score types are:'r' (Pearson's correlation coefficient), 'pp' (Predictive Power score, a Learning Tree based dependence score based on library 'ppscore') or 'mi' (Normalized mutual information)")
        return score
        


    def feature_importance(self,
        Y:typing.Union[str,typing.List[str],pd.Series,pd.DataFrame],
        target_type:str,
        score_type:str,
        threshold=None, ## either None (all features returned), int n (first n features rturned) or float f in [0,1] (first n features where n corresponds to a fraction f of the total number of features)
        y_threshold=None, ## either positional index of target or name of target to use to order 
        **kwargs
    ):
        ## validation
        if type(Y) is pd.Series:
            if not self._df.index.equals(Y.index):
                raise Exception("when 'Y' is passed as an external pandas.Series instance, it must have the same index as the pandas.DataFrame instance the method is called on")
            Y = Y.to_frame()
            X = self._df
        if type(Y) is pd.DataFrame:
            if not self._df.index.equals(Y.index):
                raise Exception("when 'Y' is passed as an external pandas.DataFrame instance, it must have the same index as the pandas.DataFrame instance the method is called on")
            X = self._df
        elif type(Y) is str:
            Y=self._df[[Y]]
            X = self._df.drop(
                Y,
                axis = 1
            )
        elif type(Y) is list and all([type(t) is str for t in Y]):
            Y=self._df[Y]
            X = self._df.drop(
                Y,
                axis = 1
            )
        else:
            raise Exception("'Y' must be either a string, a list of strings, a pandas.Series instance or a pandas.DataFrame instance")


            


        scores = []
        for y_name,y in tqdm(list(Y.iteritems())):
            scores.append(
                X.apply(
                    lambda x: X.feature_importance.get_score(
                        x.name,
                        y,
                        target_type,
                        score_type,
                        **kwargs
                    )
                )
            )
        scores = pd.concat(
            objs=scores,
            axis=1,
            join='outer',
            keys=Y.columns
        )

        if y_threshold is None:
            y_threshold = Y.columns[0]
        elif type(y_threshold) is not str:
            raise Exception("'y_threshold' must be a string naming the name of the target to use to order scores")

        if threshold is None:
            indexer = scores.index
        elif threshold<1 and threshold>0:
            indexer = scores[y_threshold]>threshold
        elif not threshold>=1:
            raise Exception('threshold must be either a float between 0 and 1 or an integer above 1')
        
        return scores.sort_values(
            by=y_threshold
        ).loc[indexer]
    
    def dependence_matrix(self,
        score_type:str
    ):
        matrix = pd.DataFrame()
        for i,x in tqdm(list(enumerate(self._df.columns))):
            matrix = matrix.join(
                self._df.iloc[
                    :,
                    i:
                ].feature_importance.feature_importance(
                    Y=self._df.iloc[
                        :,
                        i:
                    ],
                    target_type='reg',
                    score_type=score_type,
                )
            )










    # def cross_corr(y, x, max_lag = 24, measure = 'Pearson', return_callable = False):
    #     l = y.shape[0]
    #     if measure == 'Pearson':
    # #     x = (np.array(x) - np.mean(x))/np.std(x)
    # #     y = (np.array(y) - np.mean(y))/np.std(y)
    #         cc = [np.corrcoef(y[lag:],x[:(l-lag)])[1,0] for lag in range(min(max_lag+1, l))]
    #     elif measure == 'PPS':
    #         cc = []
    #         for lag in range(min(max_lag+1, l)):
    #             df = pd.DataFrame()
    #             df['x'] = x[:(l-lag)]
    #             df['y'] = y[lag:]
    #             cc.append(
    #                 pps.score(df,'x','y')['ppscore']
    #             )
    #     if return_callable:
    #         return (lambda lag: cc[lag])
    #     return cc






    # def plot_cross_corr(cc, title = None, **kwargs):
    #     fig = plt.figure(figsize = (10,3))
    #     plt.plot(cc,'ob', **kwargs)
    #     plt.title(title)
    #     plt.vlines(range(len(cc)), [0], cc)
    #     plt.show()



    # def plot_cross_corr2(cc_train, cc_test, title_train = None, title_test = None, vline = None, **kwargs):
    #     fig,axs = plt.subplots(1,2,figsize = (20,3))
    #     axs[0].plot(cc_train,'ob', **kwargs)
    #     axs[0].set_title(title_train)
    #     axs[0].vlines(range(len(cc_train)), [0], cc_train)
    #     axs[1].plot(cc_test,'ob', **kwargs)
    #     axs[1].set_title(title_test)
    #     axs[1].vlines(range(len(cc_test)), [0], cc_test)
    #     if vline is not None:
    #         axs[0].axvline(x = vline, c = 'red')
    #         axs[1].axvline(x = vline, c = 'red')
    #     plt.show()



    # def plot_cross_corr3(ccs, titles = None, vline = None, **kwargs):
    #     #n_plots_x = np.ceil(len(ccs)/2)
    #     fig,axs = plt.subplots(7, 2, figsize = (20,21))
    #     for ax,cc,title in zip(axs.flatten(),ccs,titles):
    #         ax.plot(cc,'ob', **kwargs)
    #         ax.vlines(range(len(cc)), [0], cc)
    #         ax.set_title(title)
    #         if vline is not None:
    #             ax.axvline(x = vline, c = 'red')
    #     plt.show()





    # def filtered_cross_corr(target, feature, max_lag = 24, filter_arr = None, return_callable = False):
                        
    #     l = target.shape[0]
        
    #     ### se non viene passato nessun filtro sui dati metto a True tutti i valori di filter_arr
    #     if filter_arr is None:
    #         filter_arr = np.array([True]*l)
        
    #     lag_filter_arr = np.array([True]*l)  ### filtro booleano per escludere le osservazioni fino al lag di interesse
    # #     print('lag_filter_arr_shape: ', lag_filter_arr.shape)
    # #     print('filter_arr_shape: ', filter_arr.shape)
    # #     print('target_shape: ', target.shape)
    # #     print('feature_shape: ', feature.shape)
    # #     print('feature_shape: ', feature.shape)
    # #     print('mix_shape: ', (filter_arr & lag_filter_arr).shape)
        
    #     cc = []
    #     for lag in range(min(max_lag+1, l)):
    #         ### metto tutti i valori iniziali fino al lag voluto pari a False in modo che non vengano inclusi 
    #         lag_filter_arr[:lag] = False 

            
    #         cc.append( 
    #             np.corrcoef(
    #                 target[(filter_arr & lag_filter_arr)],   
    #                 feature.shift(lag)[(filter_arr & lag_filter_arr)] ### prima shifto la feature del lag voluto, poi filtro tramite
    #                                                         ###l'array di booleani filter_arr 
    #             )[1,0]  #corrcoef restituisce un matrice di correlazione, prendiamo solo il valore che ci interessa
    #         )
                
                
            
        
    #     if return_callable:
    #         return (lambda lag: cc[lag])
    #     return cc



