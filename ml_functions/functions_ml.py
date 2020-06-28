import pandas as pd
import numpy as np

import calendar
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import  LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN , RandomOverSampler
from imblearn.under_sampling import  RandomUnderSampler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import KFold

# regression
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error

# classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Save models
from sklearn.externals import joblib


# EDA
def percent_nulls_by_column(column):
    """
    Divide la cantidad de nulos por filas sobre el total de filas
    :return: Retorna  un diccionario de los indices de las  filas con el porcentaje de nulos por fila
    """
    print("percentage_nulls_by: ", column.name)
    nulls_per_row = column.isnull().sum(axis=0)
    ind = nulls_per_row / len(column) * 100
    return ind

def col_with_outliers(column):
    print("col_with_outliers")
    q1 = np.nanpercentile(column, 25)
    q3 = np.nanpercentile(column, 75)
    ri = q3 - q1
    # Valores extremos estan fuera de [q1-1.5*ri,q3+1.5*ri] -> Diagrama de
    # Cajas
    a = q1 - 1.5 * ri
    b = q3 + 1.5 * ri

    outliers = [True  if i < a.item() or b.item() < i else False for i in column]
    index_outliers = column.index[outliers]
    return str(index_outliers.values)

def rows_high_percentage_nulls(dataframe,MAX_PER_NULL_ROW = 0.3, MAX_PER_NULL_DATASET = 0.4):
    """
    Divide la cantidad de nulos por filas sobre el total de filas
    :param dataframe:
    :type dataframe: Pandas Dataframe object
    :return: true o false
    """
    print("rows_high_percentage_nulls")
    length_rows = len(dataframe)
    length_cols = len(dataframe.columns)
    nulls_per_row = dataframe.isnull().sum(axis=1)
    ind = nulls_per_row / length_cols
    count = 0
    for value in ind:
        if value >= float(MAX_PER_NULL_ROW):
            count+=1
        else:
            continue

    per_dat_null = count/length_rows

    if per_dat_null >= float(MAX_PER_NULL_DATASET):
        return True
    else:
        return False

def num_duplicate_row(dataframe):
    """
    :param dataframe:
    :type dataframe: Pandas Dataframe object
    :return: num of duplicate rows
    """
    print("num_duplicate_row")
    series = dataframe.duplicated(keep = False)
    return series.sum()

def get_attributes(data,list):
    """
    :param dataframe_cols:
    :type dataframe_cols:
    """
    names = dict()
    print("count_attribute_numeric")

    count = 0

    for name, type_col in zip(data.columns, data.dtypes):
        if str(type_col) not in ["object","category"]:
            count+=1
            names[name].append(name)
        else:
            continue
    return count, names

def count_attribute_categoric(data):
    """
    :param dataframe_cols:
    :type dataframe_cols:
    """
    print("count_attribute_categoric")
    count = 0

    for type_col in dataframe_cols:
        if str(type_col) in [TYPE_CATEGORIC,TYPE_OBJECT]:
            count+=1
        else:
            continue
    return count


def plot_categorical(data):
    list_cat = data.columns
    if (len(list_cat) / 3) <=1:
        fig, axes = plt.subplots(len(data.columns), 1,figsize=(12,12))

        for i, ax in enumerate(fig.axes):
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            sns.countplot(x=data.columns[i], alpha=0.7, data=data)
    else :

        fig, axes = plt.subplots(round(len(list_cat) / 3), 3, figsize=(12, 30))

        for i, ax in enumerate(fig.axes):
            if i < len(list_cat):
                ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
                sns.countplot(x=data.columns[i], alpha=0.7, data=data, ax=ax)

def info_date(series_date):
    date_dictionary = dict()
    date_dictionary["n_years"] = series_date.dt.year.unique()
    date_dictionary["date_max"] = series_date.max()
    date_dictionary["date_min"] = series_date.min()
    date_dictionary["cantidad_dias"] = ((series_date.max() - series_date.min()).days + 1)
    dataframes = []
    for year in date_dictionary["n_years"]:
        date_dictionary["registros_"+str(year)] = len(series_date[series_date.dt.year==year])
        meses = []
        months = []
        cantidades = []
       
        for i in range(1,13):
            month = calendar.month_name[i]
            months.append(month)
            cantidad = len(series_date[series_date==(str(year)+"-"+str(i))])
            cantidades.append(cantidad)
            # meses.append((month, cantidad))

        dtime = pd.DataFrame({'year':year,'mes':months,'cantidad':cantidades})
        dataframes.append(dtime)
    
        # date_dictionary[str(year)+"_meses"] = meses
    data = pd.concat(dataframes)
    g = sns.catplot(x="mes", y="cantidad", data=data, col="year", kind="bar")
    loc, labels = plt.xticks()
    g.set_xticklabels(labels, rotation=30)
    # g = sns.FacetGrid(data, col="year")
    plt.plot()

    return date_dictionary


def date_target_behaviour(data, label_date, label_target, type="classification", measure="mean", paleta= "colorblind"):
    date_dictionary = dict()
    info = data.groupby([data[label_date].dt.year,data[label_date].dt.month],as_index=False)[label_target].agg([measure])
    info.index.names = ["year","mes"]
    info.columns = [measure]
    info.reset_index(level=['year', 'mes'], inplace=True)

    info["mes"] = info["mes"].transform(lambda x: calendar.month_name[x])
    current_palette_7 = sns.color_palette(paleta, len(info.year.unique()))

    g = sns.lineplot(x= "mes", y = measure, data=info, hue="year",palette= current_palette_7)
    labels=info.mes.unique()
    g.set_xticklabels(labels, rotation=30)
    plt.plot()


def convert_dummies(data, name):
    encoder = LabelBinarizer()
    dummies = encoder.fit_transform(data[name].values.reshape(-1,1))
    name = data[name].name
    df_dummies = pd.DataFrame(dummies)
    names = []
    for i in encoder.classes_:
    	names.append(i+name)
    df_dummies.columns = names
    data.drop(name, axis=1, inplace=True)
    data = pd.concat([data, df_dummies], axis=1)
    return data

def outliers_percentage(column):
    """
    Los Valores extremos que estan fuera de [q1-1.5*ri,q3+1.5*ri] seran considerados outliers
    :return: Retorna el porcentaje de outliers por columna
    """
    print("outliers_percentage")
    q1 = np.nanpercentile(column, 25)
    q3 = np.nanpercentile(column, 75)
    ri = q3 - q1

    # Valores extremos estan fuera de [q1-1.5*ri,q3+1.5*ri] -> Diagrama de
    # Cajas
    a = q1 - 1.5 * ri
    b = q3 + 1.5 * ri

    outliers1 = [1 for i in column if i < a.item() or b.item() < i]

    sum = np.sum(outliers1)

    return sum / len(column)

# Preprocessing
def likelihood_encoding(data,var_cat, name_target):
	for var in var_cat:
		data[var+'likelihoood'] = data.groupby(var)[name_target].transform(lambda x: x.mean())
	return data


def impact_coding(data, feature, target='y'):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.
    
    In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    could be moved to a parameter.
    '''
    n_folds = 3
    n_inner_folds = 2
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    print("-----------------------")
    print(impact_coded.head())
    print("-----------------------")
    print(oof_mean_cv.mean(axis=1))
    print("-----------------------")
    print(oof_default_mean)
    print("-----------------------")

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean


def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    # recorre toda la lista attr y extrae los atributos definidos a traves de getattr
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)



# Validation
def model_comparison():
    pass



def experiment_model(algorithm_obj, X, y, X_test, metrics, model_params, name_algorithm ="", type_model = "sklearn",
    dataset_version="default", parameters="default", type_problem ="classification", 
    type_validation="kfolds", cv_folds=10, save_model=False, name_model="model", plot_fimportance=False):
    
    metrics_collected = []
    if type_problem =="classification":
        cols = ['dataset', 'algoritmo', 'params', 
                                'valid_logloss', 'valid_logloss_std',
                                'train_logloss', 'train_logloss_std',
                                'logloss_diff',

                                'accuracy_valid','accuracy_valid_std',
                                'accuracy_train','accuracy_train_std'
                                'accuracy_diff'

                                'recall_valid','recall_valid_std'
                                'recall_train','recall_train_std'
                                'recall_diff',
                                'time']

    else :
        cols = ['dataset', 'algoritmo', 'params', 
                                'valid_rsme', 'valid_rsme_std',
                                'train_rsme', 'train_rsme_std',
                                'rsme_diff',
                                'time']

    tabla_resultados =pd.DataFrame(columns=cols)

    # **************************************************************************************************************
    if type_validation == "cv":
        cross_obj = cross_validate(algorithm_obj, X, y, cv=cv_folds, scoring=metrics,  return_train_score=True)
        print(cross_obj)



    # **************************************************************************************************************
    elif type_validation=="holdout": 
        # HoldOut
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.3 )
        model_trained =  algorithm_obj.fit(X_train, y_train)

        if save_model == True:
            joblib.dump(model, name+'.pkl') 

        y_predicted = model_trained.predict(X_test)

        for metric in metrics:
            value = eval(metric+'(y_test,y_predicted)')
            tabla_resultados[metric] = value
            print((metric,value))

        print(metrics_collected)

    # **************************************************************************************************************
    elif type_validation == "kfolds":
        start = time.time()
        valid_logloss = []
        train_logloss = []
        valid_accuracy = []
        train_accuracy = []
        valid_recall = []
        train_recall = []

        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kfolds.split(X, y):

            if type_model == "lightgbm":

                model = algorithm_obj.fit(X.iloc[train_idx,:], y[train_idx], 
                eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
                eval_metric='log_loss', 
                early_stopping_rounds=30)

                y_predicted = model.predict(X.iloc[test_idx,:])
                print(confusion_matrix(y.iloc[test_idx], y_predicted))



            elif type_model =="sklearn":
                model = algorithm_obj.fit(X.iloc[train_idx,:], y[train_idx])

                y_predicted = model.predict(X.iloc[test_idx,:])
                print(confusion_matrix(y.iloc[test_idx], y_predicted))



            # Test prediction metricas
            valid_logloss += [log_loss(y.iloc[test_idx], y_predicted)]
            valid_accuracy += [accuracy_score(y.iloc[test_idx], y_predicted)]
            valid_recall += [recall_score(y.iloc[test_idx], y_predicted)]



            y_predicted_train = model.predict(X.iloc[train_idx,:])
            # Train predictions metricas
            train_logloss += [log_loss(y.iloc[train_idx], y_predicted_train)]
            train_accuracy += [accuracy_score(y.iloc[train_idx], y_predicted_train)]
            train_recall += [recall_score(y.iloc[train_idx], y_predicted_train)]


            end = time.time()

            tabla_resultados.loc[len(tabla_resultados)] = [dataset_version, name_algorithm, parameters,
                                         np.mean(valid_logloss), np.std(valid_logloss), 
                                         np.mean(train_logloss), np.std(train_logloss),
                                         np.mean(valid_logloss) - np.mean(train_logloss),


                                         np.mean(valid_accuracy), np.std(valid_accuracy), 
                                         np.mean(train_accuracy), np.std(train_accuracy),
                                         np.mean(valid_accuracy) - np.mean(train_accuracy),

                                         np.mean(valid_recall), np.std(valid_recall), 
                                         np.mean(train_recall), np.std(train_recall),
                                         np.mean(valid_recall) - np.mean(train_recall) ,
                                         round(end - start, 2)]
        

    return y_test, tabla_resultados

# Balancear datos
def oversampling(X, y):
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_sample(X, y)
    return X_resampled, y_resampled

def undersampling(X, y):
    rus = RandomUnderSampler(random_state=0, replacement=True)
    X_resampled, y_resampled = rus.fit_sample(X, y)
    return X_resampled, y_resampled

def smote(X, y):
    sm = SMOTE(random_state=42, ratio='minority')
    X_res, y_res = sm.fit_sample(X, y)
    return X_res, y_res

def adasyn(X,y):
    ada = ADASYN()
    X_resampled, y_resampled = ada.fit_sample(X, y)
    return X_resampled, y_resampled


def randomforest_classifier(**parameters):
    rf = RandomForestClassifier()
    if len(parameters.items())>0:
        rf.set_params(**parameters)
    return rf

# rank features









