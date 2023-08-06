# Loading necessary Libraries
import json
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import hvplot.pandas
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from statsmodels.formula.api import ols
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt  # To visualize
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

# Deep Learning libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
import keras
from keras.callbacks import EarlyStopping
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 

from tensorflow import keras
import keras_tuner as kt

# Defining target and features
target = "module_temperature"
feature_names = ['irradiance', 'wind_speed', 'wind_direction', 'ambient_temperature']


# Read CSv file and Return dataframe given module ID from the user
def get_csv(user_input):
    pathname = "Data/test_samples/" + user_input + ".csv"
    df = pd.read_csv(Path(pathname), header=1, usecols = ['TIMESTAMP', 'AirTemp_Avg', 'Irradiance_Pyr_NOCT_Avg', 'Wind_speed', \
                                                          'Wind_direction', 'NOCT_Mod1_Cent1_Avg', 'NOCT_Mod1_Cent2_Avg'])
    return df


# This function takes a dataframe and split into a list of dataframes to obtain daily data
def split_df(df):
    df_lst = []  # declare the list to return
    df['date'] = df['date'].astype(str)  # convert date column to string
    dates = df['date'].unique().tolist() # store unique dates into a list
    for Date in dates:
        df_lst.append(df.loc[df['date'].str.contains(Date)])   # create and append df of rows that contain the date string
        
    return df_lst


# Create a deep neural network model with two hidden layers
def create_model():
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=4))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(4, activation="relu"))
    
    # Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
    # Typically ReLu-based activation are used but since it performs regression, it needs a linear activation.
    model.add(Dense(1, activation="linear"))
    
    # Compile model: The model is initialized with the Adam optimizer and then it is compiled.
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
    
    return model


# Fit the DNN model with early stopping
def fit_model(model, monitor, epoch):
    # Patient early stopping
    es = EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=200)
    
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=100, verbose=2, callbacks=[es])
    
    return history

# Saving the model
def saving_model(model, filename):
    # Save a model using the HDF5 format
    model.save(filename)
    

# Saving history
def saving_history(history, filepath):
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(filepath, 'w'))
    
# Load a model from the HDF5 format
def loading_model(filename):
    loaded_h5_model = tf.keras.models.load_model(filename)
    
    return loaded_h5_model

# Load history from the json format
def loading_history(filename):
    history_dict = json.load(open(filename, 'r'))
    
    return history_dict

# Calculate predictions
def calculate_predictions(model, X_train, X_test):
    predictions = {}
    predictions['train'] = model.predict(X_train)
    predictions['test'] = model.predict(X_test)
    
    return predictions

# Plot training history
def plot_training(history_dict, filepath):
    plt.plot(history_dict['loss'], label='train')
    plt.plot(history_dict['val_loss'], label='test')
    plt.legend()
    plt.savefig(filepath)
    
# Plot actual vs prediction
def actual_prediction_plot(y, prediction_set, title, filepath):
    plt.plot(y,prediction_set,'ro')
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(filepath)
    
# Compute scores value for a set
def compute_metrics(y, prediction_set):
    scores = []
    scores.append(metrics.r2_score(y,prediction_set))
    scores.append(metrics.mean_squared_error(y,prediction_set))
    
    return scores

# Compute NOCT value
def compute_noct(model):
    x = pd.DataFrame(columns=['irradiance', 'wind_speed', 'wind_direction', 'ambient_temperature'])
    x.loc[0] = [800,1,120,20]
    x = np.asarray(x)
    x=tf.convert_to_tensor(x, dtype=tf.int32)
    y_pred = model.predict(x)
    noct = float("{:.2f}".format( y_pred[0][0] ))
    
    return noct



# Pre-processing
# This function takes a newly loaded data, which is a csv files of multiple days collection
# and returns a pre-processed dataframe of multiple days data
def preprocessing(combined_data):
    # Drop rows.
    combined_data.drop(combined_data.index[[0,1]], axis=0, inplace=True)
    
    # Change "NAN" values to NaN so they can be dropped
    combined_data.replace("NAN", np.NaN, inplace=True)
    
    # Drop the previous index and reset 
    combined_data.reset_index(drop=True, inplace=True)
    
    # Convert numeric columns to float
    data = ['AirTemp_Avg', 'Irradiance_Pyr_NOCT_Avg', 'Wind_speed', 'Wind_direction', 'NOCT_Mod1_Cent1_Avg', 'NOCT_Mod1_Cent2_Avg']
    for col in data:
        combined_data[col] = combined_data[col].astype('float')
        
    
    # Delete all data outside the range 0.25 < wind speed < 1.75
    indexWS = combined_data[ (combined_data['Wind_speed']<0.25) | (combined_data['Wind_speed']>1.75) ].index  # get index of the desired rows to drop
    combined_data.drop(indexWS, axis=0, inplace=True)
    
    # Delete all Wind Direction data within 290 - 250, and 69 - 110
    indexWD = combined_data[ ((combined_data['Wind_direction']>250) & (combined_data['Wind_direction']<=290)) \
                            | ((combined_data['Wind_direction']>69.99) & (combined_data['Wind_direction']<=110)) ].index
    combined_data.drop(indexWD, axis=0, inplace=True)
    
    # Delete all Irradiance data below 400 and above 1100
    indexIrr = combined_data[ (combined_data['Irradiance_Pyr_NOCT_Avg']<400) | (combined_data['Irradiance_Pyr_NOCT_Avg']>1100) ].index
    combined_data.drop(indexIrr, axis=0, inplace=True)
    
    # Create the T-rise and module_temperature columns
    combined_data['module_temperature'] = combined_data[['NOCT_Mod1_Cent1_Avg','NOCT_Mod1_Cent2_Avg']].mean(axis=1)
    combined_data['T_RISE'] = combined_data[['NOCT_Mod1_Cent1_Avg','NOCT_Mod1_Cent2_Avg']].mean(axis=1) - combined_data['AirTemp_Avg']
    
    # Processing timestamp/dates
    combined_data["Date"] = pd.to_datetime(combined_data["TIMESTAMP"])
    combined_data["date"] = combined_data["Date"].dt.date
    combined_data["year"] = combined_data["Date"].dt.year
    combined_data["month"] = combined_data["Date"].dt.month
    combined_data["day"] = combined_data["Date"].dt.day
    combined_data["hour"] = combined_data["Date"].dt.hour
    combined_data["minute"] = combined_data["Date"].dt.minute
    
    combined_data.dropna(axis=0, how='any', inplace=True)
    combined_data.sort_values(by=['Date'])
    
    # Slice data between 11am and 1pm
    indexHour = combined_data[ ((combined_data['hour']<11) | (combined_data['hour']>13))].index
    combined_data.drop(indexHour, axis=0, inplace=True)
    
    # Split the data into a list of daily data
    noct_dataframes = split_df(combined_data)
    
    # Aggregate data by minutes
    daily_noct = []
    for dataframe in noct_dataframes:
        noct_df = dataframe[["date", "hour","minute", "month", "AirTemp_Avg", "Wind_speed", \
                             "Wind_direction", "Irradiance_Pyr_NOCT_Avg", "module_temperature", "T_RISE"]]
        noct_df_agg = noct_df.groupby(["date", "hour", "minute"]).mean()
        noct_df_agg.reset_index(inplace=True)
        daily_noct.append(noct_df_agg)
        
    # Cleaning ambient temperatures
    # If min amb temp around 35, delete data where (max - min) amb temp are above 5 degC
    # Otherwise, Delete all Ambient Temperature data below 5 degC and above 35 degC if max Tamb < 35
    for noct in daily_noct:
        indexShift = noct[ (noct['AirTemp_Avg'] - noct['AirTemp_Avg'].shift(1)) < 0 ].index
        noct.drop(indexShift, axis=0, inplace=True)
        if noct['AirTemp_Avg'].min() >= 35:
            while (noct['AirTemp_Avg'].max() - noct['AirTemp_Avg'].min()) > 5:
                noct.drop(noct[(noct['AirTemp_Avg'] == noct['AirTemp_Avg'].max()) | \
                               (noct["AirTemp_Avg"] == noct['AirTemp_Avg'].min())].index, axis=0, inplace=True)
        else:
            indexTamb = noct[ ((noct['AirTemp_Avg']<5) & (noct['AirTemp_Avg']>35))].index
            noct.drop(indexTamb, axis=0, inplace=True)

    
    # We want to delete all the data which do not have points before and after noon
    to_remove = []   # This will hold the indices of the dataframe list to be removed
    for i, noct in enumerate(daily_noct):
        am = False
        pm = False
        if noct['hour'].min() == 11:
            am = True
        if noct['hour'].max() in [12,13]:
            pm = True
        if (am == False or pm == False):
            to_remove.append(i)
    
    # Sort the indice of dataframes to drop
    # By starting from higher to lower indices, we're guaranteed that the indices on the dataframe will remain unmodified while deleting
    for x in sorted(to_remove, reverse=True):
        del daily_noct[x]
        
    
    # Combining the list of dataframes into a single one and return
    combined_daily_noct = pd.concat(daily_noct, axis=0, join='inner')
    
    # Renaming some columns
    combined_df = combined_daily_noct[['AirTemp_Avg', 'Wind_speed', 'Wind_direction', 'Irradiance_Pyr_NOCT_Avg', 'module_temperature', 'T_RISE']]
    combined_df.rename(columns={'AirTemp_Avg':'ambient_temperature', 'Wind_speed':'wind_speed', \
                                'Wind_direction':'wind_direction', 'Irradiance_Pyr_NOCT_Avg':'irradiance'}, inplace=True)
    
    
    return combined_df


# Multiple Linear Regressions using LinearRegression() estimator
def multiple_linear_regression(combined_df, user_input):
    mlr = {}  # for storing the linear model parameters
    
    # define filepath for plots
    plotpath = 'plots/dnn_noct_' + user_input
    
    sample_df = combined_df.sample(frac=0.1, random_state=17)
    
    # Plotting the features
    for feature in feature_names:
        plt.figure(figsize=(16,9))
        sns.scatterplot(data=sample_df, x=feature, y=target, hue=target, palette='cool', legend=False)
        
    # split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(combined_df[feature_names], combined_df[target], random_state=11)
    
    # Instantiate and fit a regression model
    linear_regression = LinearRegression()
    linear_regression.fit(X=X_train, y=y_train)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    
    # Testing the model
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    # Visualizing the Expected vs. Predicted
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)
    figure = plt.figure(figsize=(9,9))
    axes = sns.scatterplot(data=df, x= 'Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)

    # Set the x- and y-axes limits to use the same scale along both axes
    start = min(expected.min(), predicted.min())
    end = max(expected.max(), predicted.max())
    axes.set_xlim(start, end)
    axes.set_ylim(start, end)
    line = plt.plot([start, end], [start, end], 'k--')
    plt.savefig(plotpath+"exp_pred_mlr")
    
    # Compute the estimate
    mlr['noct'] = linear_regression.intercept_ + 800*linear_regression.coef_[0] + 1*linear_regression.coef_[1] + \
    120*linear_regression.coef_[2] + 20*linear_regression.coef_[3]
    
    # Regression Model Metrics
    mlr['r_square'] = metrics.r2_score(expected, predicted)
    mlr['mse'] = metrics.mean_squared_error(expected, predicted)
    
    return mlr
    
# Deep Learning
# This function takes a dataframe and a user input to create and save a deep neural network with two hidden layers
def deep_learning_save(combined_df, user_input):
    
    # define filename for saving the model and history
    filepath = 'models/dnn_noct_' + user_input
    
    # Split into training and testing windows
    X_train, X_test, y_train, y_test = train_test_split(combined_df[feature_names], combined_df[target], random_state=11)
    
    # Let's set the model training for a very long time,
    # but with an EarlyStopping callback so it stops automatically when it stops improving.
    model = create_model()
    history = fit_model(model, 'val_loss', 100000)
    
    # Save a model using the HDF5 format
    saving_model(model, filepath+'.h5')
    
    # Save history
    saving_history(history, filepath)
    

# Deep Learning loading
# This function takes a dataframe and a user input to create and save a deep neural network with two hidden layers
def deep_learning_load(combined_df, user_input):
    
    # Define a dictionary for storing the model parameters
    dnn = {}
    
    # Split into training and testing windows
    X_train, X_test, y_train, y_test = train_test_split(combined_df[feature_names], combined_df[target], random_state=11)
    
    # define filename for saving the model and history
    filepath = 'models/dnn_noct_' + user_input
    
    # define filepath for plots
    plotpath = 'plots/dnn_noct_' + user_input
    
    # Load a model from the HDF5 format
    model = loading_model(filepath + ".h5")
    model.summary()
    
    # Load history
    history_dict = loading_history(filepath)
    
    # Calculate predictions: returns a dictionary of keys "train" and "test"
    predictions = calculate_predictions(model, X_train, X_test)
    
    # Plot training history 
    plot_training(history_dict, plotpath + "_training")
    
    # Plot actual vs prediction for training set
    actual_prediction_plot(y_train, predictions['train'], 'Training Set', plotpath+"_train")
    
    # Plot actual vs prediction for validation set
    actual_prediction_plot(y_test, predictions['test'], 'Test Set', plotpath+"_test")
    
    # Compute score values for training set
    # Note that this returns a list of scores.The first index is r_quare and the second index is mse
    scores_train = compute_metrics(y_train, predictions['train'])
    r_square_train = scores_train[0]
    mse_train = scores_train[1]
    
    # Compute score values for training set
    scores_test = compute_metrics(y_test, predictions['test'])
    r_square_test = scores_test[0]
    mse_test = scores_test[1]
    
    # Estimate NOCT
    noct = compute_noct(model)
    
    # DNN Model Metrics
    dnn['noct'] = noct
    dnn['r_square_train'] = r_square_train
    dnn['r_square_test'] = r_square_test
    dnn['mse_train'] = mse_train
    dnn['mse_test'] = mse_test
    
    return dnn
    