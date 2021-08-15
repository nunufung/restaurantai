import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import OtherDependencies.SessionState as SessionState
from sklearn.model_selection import train_test_split
from Models.SVC import svc_param_selector
from Models.LogisticRegression import lr_param_selector
from Models.DecisionTree import dt_param_selector
from Models.RandomForet import rf_param_selector
from Models.RidgeClassifier import rc_param_selector
from Models.SGDClassifier import sgd_param_selector
from Models.KNeighborsClassifier import kn_param_selector
from Models.MLPClassifier import mlp_param_selector
from Models.RandomForestRegressor import rfreg_param_selector
from Models.KNeighborsRegressor import knreg_param_selector
from Models.DecisionTreeRegressor import dtreg_param_selector
from Models.GradientBoostingRegressor import gbreg_param_selector
from Models.MLPRegressor import mlpreg_param_selector
from Models.kerasSequential import Sequential_param_selector
from Models.kerasSequentialClassifier import Sequentialclf_param_selector

st.set_option('deprecation.showPyplotGlobalUse', False)

## Helper Functions
def dataset_selector():
    # Uploading Data
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.fillna(0)
        return df


def ExploreData(data_set):
    import matplotlib.pyplot as plt
    st.title('Explore Dataset')
    st.write(data_set)
    l = data_set['Restaurant Status'].value_counts()
    n = pd.DataFrame(data_set['Restaurant Status'].value_counts()).index.values
    plt.style.use('bmh')
    plt.figure(figsize=(15, 8))
    sns.barplot(n, l, palette='magma')
    plt.xticks(rotation=45)
    plt.xlabel('Status', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title("Count of Restaurant Status", fontsize=8)
    st.pyplot()

    box_df = data_set[['Restaurant Type', 'Area', 'District',
                       'base on District-# of ppl able to work>15 - 19',
                       'base on District-# of ppl able to work>20 - 24',
                       'base on District-# of ppl able to work>25 - 29',
                       'base on District-# of ppl able to work>30 - 34',
                       'base on District-# of ppl able to work>35 - 39',
                       'base on District-# of ppl able to work>40 - 44',
                       'base on District-# of ppl able to work>45 - 49',
                       'base on District-# of ppl able to work>50 - 54',
                       'base on District-# of ppl able to work>55 - 59',
                       'base on District-# of ppl able to work>60 - 64',
                       'base on District-# of ppl able to work>65+',
                       'base on Area #of worker location>15 - 19',
                       'base on Area #of worker location>20 - 24',
                       'base on Area #of worker location>25 - 29',
                       'base on Area #of worker location>30 - 34',
                       'base on Area #of worker location>35 - 39',
                       'base on Area #of worker location>40 - 44',
                       'base on Area #of worker location>45 - 49',
                       'base on Area #of worker location>50 - 54',
                       'base on Area #of worker location>55 - 59',
                       'base on Area #of worker location>60 - 64',
                       'base on Area #of worker location>65+', 'total Sales',
                       '#of transaction', 'sqf', 'mthly rent', 'Restaurant Type.1',
                       'Location Type', 'Restaurant Status']]
    from pylab import rcParams
    import matplotlib.pyplot as plt
    rcParams['figure.figsize'] = 7, 5
    for column in box_df:
        plt.figure()
        box_df.boxplot([column])
        st.pyplot()

    corr = data_set.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    st.pyplot()


def Classification_algos_selector():
    model_type = st.sidebar.selectbox(
        'Select Model',
        (
            'SVC',
            'LogisticRegression',
            'DecisionTreeClassifier',
            'RandomForestClassifier',
            'RidgeClassifier',
            'SGDClassifier',
            'KNeighborsClassifier',
            'MLPClassifier',
            'Deep_Sequential_Neural_Network')
    )
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    with model_training_container:
        if model_type == "SVC":
            model = svc_param_selector()
        elif model_type == 'LogisticRegression':
            model = lr_param_selector()
        elif model_type == 'DecisionTreeClassifier':
            model = dt_param_selector()
        elif model_type == 'RandomForestClassifier':
            model = rf_param_selector()
        elif model_type == 'RidgeClassifier':
            model = rc_param_selector()
        elif model_type == 'SGDClassifier':
            model = sgd_param_selector()
        elif model_type == 'KNeighborsClassifier':
            model = kn_param_selector()
        elif model_type == 'MLPClassifier':
            model = mlp_param_selector()
        elif model_type == 'Deep_Sequential_Neural_Network':
            model = Sequentialclf_param_selector()
    return model, model_type


def Regression_algos_selector():
    model_type = st.sidebar.selectbox(
        'Select Model',
        (
            'RandomForestRegressor',
            'KNeighborsRegressor',
            'DecisionTreeRegressor',
            'GradientBoostingRegressor',
            'MLPRegressor',
            'Neural_Network_Back_propagation')
    )
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    with model_training_container:
        if model_type == 'RandomForestRegressor':
            model = rfreg_param_selector()
        elif model_type == 'KNeighborsRegressor':
            model = knreg_param_selector()
        elif model_type == 'DecisionTreeRegressor':
            model = dtreg_param_selector()
        elif model_type == 'GradientBoostingRegressor':
            model = gbreg_param_selector()
        elif model_type == 'MLPRegressor':
            model = mlpreg_param_selector()
        elif model_type == 'Neural_Network_Back_propagation':
            model = Sequential_param_selector()
    return model, model_type


def generate_data_classification(data_set):
    final = data_set
    X = final[['Restaurant Type', 'Area', 'District',
               'base on District-# of ppl able to work>15 - 19',
               'base on District-# of ppl able to work>20 - 24',
               'base on District-# of ppl able to work>25 - 29',
               'base on District-# of ppl able to work>30 - 34',
               'base on District-# of ppl able to work>35 - 39',
               'base on District-# of ppl able to work>40 - 44',
               'base on District-# of ppl able to work>45 - 49',
               'base on District-# of ppl able to work>50 - 54',
               'base on District-# of ppl able to work>55 - 59',
               'base on District-# of ppl able to work>60 - 64',
               'base on District-# of ppl able to work>65+',
               'base on Area #of worker location>15 - 19',
               'base on Area #of worker location>20 - 24',
               'base on Area #of worker location>25 - 29',
               'base on Area #of worker location>30 - 34',
               'base on Area #of worker location>35 - 39',
               'base on Area #of worker location>40 - 44',
               'base on Area #of worker location>45 - 49',
               'base on Area #of worker location>50 - 54',
               'base on Area #of worker location>55 - 59',
               'base on Area #of worker location>60 - 64',
               'base on Area #of worker location>65+', 'total Sales',
               '#of transaction', 'sqf', 'mthly rent', 'Restaurant Type.1',
               'Location Type']]
    y = final[['Restaurant Status']]
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    return X_train, X_test, y_train, y_test


def generate_data_regression(data_set):
    X = data_set[['Restaurant Type', 'Area', 'District',
                  'base on District-# of ppl able to work>15 - 19',
                  'base on District-# of ppl able to work>20 - 24',
                  'base on District-# of ppl able to work>25 - 29',
                  'base on District-# of ppl able to work>30 - 34',
                  'base on District-# of ppl able to work>35 - 39',
                  'base on District-# of ppl able to work>40 - 44',
                  'base on District-# of ppl able to work>45 - 49',
                  'base on District-# of ppl able to work>50 - 54',
                  'base on District-# of ppl able to work>55 - 59',
                  'base on District-# of ppl able to work>60 - 64',
                  'base on District-# of ppl able to work>65+',
                  'base on Area #of worker location>15 - 19',
                  'base on Area #of worker location>20 - 24',
                  'base on Area #of worker location>25 - 29',
                  'base on Area #of worker location>30 - 34',
                  'base on Area #of worker location>35 - 39',
                  'base on Area #of worker location>40 - 44',
                  'base on Area #of worker location>45 - 49',
                  'base on Area #of worker location>50 - 54',
                  'base on Area #of worker location>55 - 59',
                  'base on Area #of worker location>60 - 64',
                  'base on Area #of worker location>65+', 'Restaurant Status',
                  '#of transaction', 'sqf', 'mthly rent', 'Restaurant Type.1',
                  'Location Type']]
    y = data_set['total Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
    return X_train, X_test, y_train, y_test


def KFold_Classify(dataset):
    X = dataset[['Restaurant Type', 'Area', 'District',
                 'base on District-# of ppl able to work>15 - 19',
                 'base on District-# of ppl able to work>20 - 24',
                 'base on District-# of ppl able to work>25 - 29',
                 'base on District-# of ppl able to work>30 - 34',
                 'base on District-# of ppl able to work>35 - 39',
                 'base on District-# of ppl able to work>40 - 44',
                 'base on District-# of ppl able to work>45 - 49',
                 'base on District-# of ppl able to work>50 - 54',
                 'base on District-# of ppl able to work>55 - 59',
                 'base on District-# of ppl able to work>60 - 64',
                 'base on District-# of ppl able to work>65+',
                 'base on Area #of worker location>15 - 19',
                 'base on Area #of worker location>20 - 24',
                 'base on Area #of worker location>25 - 29',
                 'base on Area #of worker location>30 - 34',
                 'base on Area #of worker location>35 - 39',
                 'base on Area #of worker location>40 - 44',
                 'base on Area #of worker location>45 - 49',
                 'base on Area #of worker location>50 - 54',
                 'base on Area #of worker location>55 - 59',
                 'base on Area #of worker location>60 - 64',
                 'base on Area #of worker location>65+', 'total Sales',
                 '#of transaction', 'sqf', 'mthly rent', 'Restaurant Type.1',
                 'Location Type']]
    y = dataset[['Restaurant Status']]
    from sklearn.model_selection import KFold
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics
    kf = KFold(n_splits=3)
    kf.get_n_splits(X)
    print(kf)
    KFold(n_splits=5, random_state=None, shuffle=False)
    slider_container = st.sidebar.beta_expander("Train a model", True)
    with slider_container:
        clf = rf_param_selector()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.50,shuffle=False)
        clf.fit(X_train, y_train.values.ravel())
        y_true, y_pred = y_test, clf.predict(X_test)
        train_score = int(clf.score(X_train, y_train) * 100)
        test_score = int(100 * clf.score(X_test, y_test))
        st.write('Train Acc:', train_score)
        st.write('Test Acc:', test_score)
        TrueResultslat = y_test
        PredictedResultslat = clf.predict(X_test)
        data = {'y_Actual': y_test['Restaurant Status'],
                'y_Predicted': PredictedResultslat
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrixf = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

        sns.heatmap(confusion_matrixf, annot=True)
        st.pyplot()
        st.write("Random_Forest Matrix: SVM ")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:Random Forest Classifier ")
        st.write(classification_report(y_test, PredictedResultslat))
        LR_accuracy = metrics.accuracy_score(y_test, PredictedResultslat)
        st.write("Random Forest Classifier Accuracy:", LR_accuracy * 100)


def KFold_Regressor(dataset):
    X = dataset[['Restaurant Type', 'Area', 'District',
                 'base on District-# of ppl able to work>15 - 19',
                 'base on District-# of ppl able to work>20 - 24',
                 'base on District-# of ppl able to work>25 - 29',
                 'base on District-# of ppl able to work>30 - 34',
                 'base on District-# of ppl able to work>35 - 39',
                 'base on District-# of ppl able to work>40 - 44',
                 'base on District-# of ppl able to work>45 - 49',
                 'base on District-# of ppl able to work>50 - 54',
                 'base on District-# of ppl able to work>55 - 59',
                 'base on District-# of ppl able to work>60 - 64',
                 'base on District-# of ppl able to work>65+',
                 'base on Area #of worker location>15 - 19',
                 'base on Area #of worker location>20 - 24',
                 'base on Area #of worker location>25 - 29',
                 'base on Area #of worker location>30 - 34',
                 'base on Area #of worker location>35 - 39',
                 'base on Area #of worker location>40 - 44',
                 'base on Area #of worker location>45 - 49',
                 'base on Area #of worker location>50 - 54',
                 'base on Area #of worker location>55 - 59',
                 'base on Area #of worker location>60 - 64',
                 'base on Area #of worker location>65+', 'Restaurant Status',
                 '#of transaction', 'sqf', 'mthly rent', 'Restaurant Type.1',
                 'Location Type']]
    y = dataset['total Sales']
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    KFold(n_splits=5, random_state=None, shuffle=False)
    KFold(n_splits=2, random_state=None, shuffle=False)
    slider_container = st.sidebar.beta_expander("Train a model", True)
    with slider_container:
        reg = knreg_param_selector()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        reg.fit(X_train, y_train)
        pre = reg.predict(X_test)

        rms = r2_score(y_test, pre)
        st.write("regression score function: ", rms)
        rms = mean_squared_error(y_test, pre)
        st.write("Mean Squared Error: ", rms)
        plt.rcParams["figure.figsize"] = (15, 6)
        plt.plot(y_test.values)
        plt.plot(pre)
        plt.title("KNeighborsRegressor total sales Predictions Test")
        plt.ylabel("Position total sales")
        plt.xlabel("Test data points")
        plt.legend(['Actual total sales', 'Predicted total sales'], loc='upper left')
        st.pyplot()


## Main 2 funtions

def sidebar_controllers():
    dataset = dataset_selector()
    selection = st.sidebar.selectbox(
        'Options',
        ('Explore Data',
         'Classification Algorithms',
         'Regression Algorithms',
         'KFold_Random_Forest_Classifier',
         'KFold_KNeighbors_Regressor')
    )
    if selection == 'Explore Data':
        if dataset is None:
            st.title('Dataset Not Found!')
            st.subheader('Please Upload Dataset! i.e, .CSV')
        else:
            ExploreData(dataset)
    if selection == 'Classification Algorithms':
        model, model_type = Classification_algos_selector()
        X_train, X_test, y_train, y_test = generate_data_classification(dataset)
        return (selection, model, model_type, X_train, X_test, y_train, y_test)
    if selection == 'Regression Algorithms':
        model, model_type = Regression_algos_selector()
        X_train, X_test, y_train, y_test = generate_data_regression(dataset)
        return (selection, model, model_type, X_train, X_test, y_train, y_test)
    if selection == 'KFold_Random_Forest_Classifier':
        KFold_Classify(dataset)
    if selection == 'KFold_KNeighbors_Regressor':
        KFold_Regressor(dataset)


def body(selection, model, model_type, X_train, X_test, y_train, y_test):
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn import metrics
    import plotly.express as px
    Clsf_algos_accuracy=pd.DataFrame(columns = ['Algorithm', 'Accuracy_(%)'])
    reg_algos_accuracy=pd.DataFrame(columns = ['Algorithm', 'Mean_Squared_Error'])
    session_state = SessionState.get(df=Clsf_algos_accuracy)
    session_state_reg = SessionState.get(df=reg_algos_accuracy)
    if selection == 'Classification Algorithms':
        if model_type == 'Deep_Sequential_Neural_Network':
            model.fit(X_train, y_train.values.ravel(), epochs=150, batch_size=10)
            # evaluate the keras model
            accuracy = model.evaluate(X_test, y_test)
            Deep_Sequential_Neural = accuracy[1] * 100
            st.subheader(model_type + ' results')
            st.write('Accuracy of Deep Sequential Neural Network: %.2f' % (Deep_Sequential_Neural))
            bool_var = 0
            for index, row in session_state.df.iterrows():
                if session_state.df.loc[index,'Algorithm'] == str(model_type):
                    session_state.df.loc[index,'Accuracy_(%)'] = Deep_Sequential_Neural
                    bool_var = 1
            if bool_var == 0:
                session_state.df = session_state.df.append({'Algorithm':model_type,'Accuracy_(%)':Deep_Sequential_Neural}, ignore_index=True)
            session_state.df=session_state.df.sort_values(['Accuracy_(%)'])
            fig = px.bar(session_state.df, x='Algorithm', y='Accuracy_(%)',title="Accuracy of each Classifier for Resturant Status Recommendation",color='Accuracy_(%)')
            st.plotly_chart(fig)
        else:
            model.fit(X_train, y_train.values.ravel())
            y_true, y_pred = y_test, model.predict(X_test)
            train_score = int(model.score(X_train, y_train) * 100)
            test_score = int(100 * model.score(X_test, y_test))
            st.subheader(model_type + ' results')
            st.write('Train Acc:', train_score)
            st.write('Test Acc:', test_score)

            ##Confusion matrix
            data = {'y_Actual': y_test['Restaurant Status'].values,
                    'y_Predicted': y_pred}
            df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
            confusion_matrixf = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'],
                                            colnames=['Predicted'])
            sns.heatmap(confusion_matrixf, annot=True)
            st.pyplot()
            st.write(model_type + " Matrix: SVM ")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:SVM ")
            st.write(classification_report(y_test, y_pred))
            LR_accuracy = metrics.accuracy_score(y_test, y_pred)
            st.write(model_type + " Classifier Accuracy:", LR_accuracy * 100)
            LR_accuracy = LR_accuracy*100
            bool_var = 0
            for index, row in session_state.df.iterrows():
                if session_state.df.loc[index,'Algorithm'] == str(model_type):
                    session_state.df.loc[index,'Accuracy_(%)'] = LR_accuracy
                    bool_var = 1
            if bool_var == 0:
                session_state.df = session_state.df.append({'Algorithm':model_type,'Accuracy_(%)':LR_accuracy}, ignore_index=True)
            session_state.df=session_state.df.sort_values(['Accuracy_(%)'])
            fig = px.bar(session_state.df, x='Algorithm', y='Accuracy_(%)',title="Accuracy of each Classifier for Resturant Status Recommendation",color='Accuracy_(%)')
            st.plotly_chart(fig)

    elif selection == 'Regression Algorithms':
        if model_type == 'Neural_Network_Back_propagation':
            model.fit(X_train, y_train, epochs=1000, verbose=0)
            ynew = model.predict(X_test)
            predicted_NNFB = ynew
            rms = mean_squared_error(ynew, y_test)
            st.subheader(model_type + ' results')
            st.write("Root Mean Square Error: ", rms)
            # Figure
            plt.rcParams["figure.figsize"] = (15, 6)
            plt.plot(y_test.values)
            plt.plot(predicted_NNFB)
            plt.title(model_type + " total sales Predictions Test")
            plt.ylabel("Total sales")
            plt.xlabel("Test data points")
            plt.legend(['Actual total sales', 'Predicted  total sales'], loc='upper left')
            st.pyplot()
            bool_var = 0
            for index, row in session_state.df.iterrows():
                if session_state_reg.df.loc[index,'Algorithm'] == str(model_type):
                    session_state_reg.df.loc[index,'Mean_Squared_Error'] = rms
                    bool_var = 1
            if bool_var == 0:
                session_state_reg.df = session_state_reg.df.append({'Algorithm':model_type,'Mean_Squared_Error':rms}, ignore_index=True)
            session_state_reg.df=session_state_reg.df.sort_values(['Mean_Squared_Error'])
            fig = px.bar(session_state_reg.df, x='Algorithm', y='Mean_Squared_Error',title="Mean Squared Error for Each Algorithm to predict sales revenue",color='Mean_Squared_Error')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig)
        else:
            model.fit(X_train, y_train)
            pre = model.predict(X_test)
            rms = r2_score(y_test, pre)
            st.subheader(model_type + ' results')
            st.write("regression score function: ", rms)
            rms = mean_squared_error(y_test, pre)
            st.write("Mean Squared Error: ", rms)
            # Figure
            plt.rcParams["figure.figsize"] = (15, 6)
            plt.plot(y_test.values)
            plt.plot(pre)
            plt.title(model_type + " total sales Predictions Test")
            plt.ylabel("Total sales")
            plt.xlabel("Test data points")
            plt.legend(['Actual total sales', 'Predicted  total sales'], loc='upper left')
            st.pyplot()
            bool_var = 0
            for index, row in session_state.df.iterrows():
                if session_state_reg.df.loc[index,'Algorithm'] == str(model_type):
                    session_state_reg.df.loc[index,'Mean_Squared_Error'] = rms
                    bool_var = 1
            if bool_var == 0:
                session_state_reg.df = session_state_reg.df.append({'Algorithm':model_type,'Mean_Squared_Error':rms}, ignore_index=True)
            session_state_reg.df=session_state_reg.df.sort_values(['Mean_Squared_Error'])
            fig = px.bar(session_state_reg.df, x='Algorithm', y='Mean_Squared_Error',title="Mean Squared Error for Each Algorithm to predict sales revenue",color='Mean_Squared_Error')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig)


if __name__ == "__main__":
    try:
        (selection, model, model_type, X_train, X_test, y_train, y_test) = sidebar_controllers()
        body(selection, model, model_type, X_train, X_test, y_train, y_test)
    except:
        pass