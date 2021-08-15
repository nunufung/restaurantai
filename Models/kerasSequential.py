import streamlit as st
from keras.models import Sequential
from keras.layers import Dense



def Sequential_param_selector():
      model = Sequential()
      model.add(Dense(20, input_dim=31, kernel_initializer='normal', activation='relu'))
      model.add(Dense(1, kernel_initializer='normal'))
      # Compile model
      loss = st.selectbox("loss", ('mse','mae'))
      optimizer = st.selectbox("optimizer", ('adam','sgd'))
      model.compile(loss= loss, optimizer=optimizer)
      return model
