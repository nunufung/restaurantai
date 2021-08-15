import streamlit as st
from keras.models import Sequential
from keras.layers import Dense



def Sequentialclf_param_selector():
      model = Sequential()
      model.add(Dense(12, input_dim=31, activation='relu'))
      model.add(Dense(8, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))
      # compile the keras model
      loss = st.selectbox("loss", ('binary_crossentropy','hinge','squared_hinge','categorical_crossentropy'))
      optimizer = st.selectbox("optimizer", ('adam','sgd'))
      model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
      return model
