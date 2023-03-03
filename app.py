#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import pickle
import pandas as pd
import plotly.express as px
import dash
from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State
import requests
import numpy as np
import prophet
import requests,lxml
#AQI Params=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone','Humidity(%)','Rainfall (mm)','windspeed (kmph)', 'winddirection_100m (°)', 'temp']
#Heatwave Params=['Rainfall (mm)','windspeed(Kmph)','Humidity(%)']
cap_hw=48.6
cap_AQI=800
pickles={0:"prophet_model_Adilabad.pkl",1:"prophet_model_Karimnagar.pkl",2:"prophet_model_Khammam.pkl",3:"prophet_model_Nizamabad.pkl",4:"prophet_model_Warangal.pkl",5:"prophet_model_AQI.pkl"}
city_codex={0:'adilabad',1:'karimnagar',2:'khammam',3:'nizamabad',4:'warangal'}
API_KEY = '9a242547bfe89eaaeca896e7c262d8cb'
dt=datetime.date.today()
dt=dt.strftime("%Y-%m-%d")
def getData(CITY_NAME):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid=9a242547bfe89eaaeca896e7c262d8cb"
    response = requests.get(url)
    jsonResponse = response.json()
    date = jsonResponse['dt']
    lat = jsonResponse['coord']['lat']
    lon = jsonResponse['coord']['lon']
    T = jsonResponse['main']['temp'] - 273.15
    min_temp = jsonResponse['main']['temp_min'] - 273.15
    max_temp = jsonResponse['main']['temp_max'] - 273.15
    humidity = jsonResponse['main']['humidity']
    wind_speed = jsonResponse['wind']['speed'] * 3.6
    wind_dir_deg = jsonResponse['wind']['deg']
    rain_mm = jsonResponse.get('rain', {}).get('1h', 0)
    weather_today=[rain_mm,wind_speed,humidity]
    weather_AQI=[humidity,rain_mm,wind_speed,wind_dir_deg,T]
    return getAQIData(CITY_NAME,weather_AQI,weather_today)

def getAQIData(CITY_NAME,weather_AQI,weather_today):
    if CITY_NAME == "warangal":
        lat = 17.968901
        lon = 79.594055
    elif CITY_NAME == "khammam":
        lat = 17.249161
        lon = 80.140007
    elif CITY_NAME == "adilabad":
        lat = 19.679430
        lon = 78.537109
    elif CITY_NAME == "nizamabad":
        lat = 18.6725
        lon = 78.0941
    elif CITY_NAME == "karimnagar":
        lat = 18.4386
        lon = 79.1288
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
#     print(response)
    jsonResponse = response.json()
    date = jsonResponse['list'][0]['dt']
    aqi = jsonResponse['list'][0]['main']['aqi']
    pm2_5 = jsonResponse['list'][0]['components']['pm2_5']
    pm10 = jsonResponse['list'][0]['components']['pm10']
    co = jsonResponse['list'][0]['components']['co']
    no2 = jsonResponse['list'][0]['components']['no2']
    so2 = jsonResponse['list'][0]['components']['so2']
    o3 = jsonResponse['list'][0]['components']['o3']
    aqi_today=[pm2_5,pm10,no2,so2,co,o3]+weather_AQI
    return weather_today,aqi_today

def tango(city):

    with open(pickles[5], 'rb') as f:
        m_aqi = pickle.load(f)
    with open(pickles[city], 'rb') as f:
        m_heat = pickle.load(f)
    weather_today,aqi_today=getData(city_codex[city])
    
    aqi_future = m_aqi.make_future_dataframe(periods=365,freq='D')
    j=0
    for i in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone','Humidity(%)','Rainfall (mm)','windspeed (kmph)', 'winddirection_100m (°)', 'temp']:
        aqi_future[i]=aqi_today[j]
        j+=1
    aqi_future['cap']=cap_AQI
    heat_future = m_heat.make_future_dataframe(periods=458,freq='D')
    j=0
    for i in ['Rainfall (mm)','windspeed(Kmph)','Humidity(%)']:
        heat_future[i]=weather_today[j]
        j+=1
    heat_future['cap']=cap_hw
    aqi_forecast=m_aqi.predict(aqi_future)
    heat_forecast=m_heat.predict(heat_future)
    aqi_forecast=aqi_forecast[['ds','yhat','yhat_lower','yhat_upper']]
    heat_forecast=heat_forecast[['ds','yhat','yhat_lower','yhat_upper']]
    return aqi_forecast,heat_forecast,city_codex[city]
#call the function
#month_data will be your required dataframe
#I suggest we use median of upper,lower and yhat values: month_data[['yhat','yhat_lower','yhat_upper']].median(axis=1) during plotting
# month=mnth#give the input of whatever month's data you want to find
# month_data=result[result['ds'].apply(lambda x:x.month==month and x.year==2023)]
#month_data will be your required dataframe
#I suggest we use median of upper,lower and yhat values: month_data[['yhat','yhat_lower','yhat_upper']].median(axis=1) during plotting
# Run the app


# In[ ]:


# Load datasets
aqi_data = pd.read_csv("aqi_data.csv")
temp_data = pd.read_csv("aqi_data.csv")
colors = {
    'background': '#E6E6FA',
    'text': '#2b044a'
}
# Initialize the app
app = dash.Dash(__name__)
global mnth
global city
mnth=1
city=0
# Define the layout
app.layout = html.Div(style={'backgroundColor': colors['background']}[
    html.H1("AQI and Temperature Graphs",style={
            'textAlign': 'center',
            'color': colors['text']
        }),
       
         html.H3('Select Month and City', style={
            'textAlign': 'center',
            'color': '#722F37'
        }),
         html.Br(),
    html.Div([
        html.Label("Month"),
        dcc.Dropdown(
            id="month-dropdown",
            options=[
                {"label": "January", "value": 1},
                {"label": "February", "value": 2},
                {"label": "March", "value": 3},
                {'label': 'April', 'value':  4},
                {'label': 'May', 'value': 5},
                {'label': 'June', 'value': 6},
                {'label': 'July', 'value': 7},
                {'label': 'August', 'value': 8},
                {'label': 'September', 'value': 9},
                {'label': 'October', 'value': 10},
                {'label': 'November', 'value': 11},
                {'label': 'December', 'value': 12}
            ],
            value=1
        ),
    ], style={"width": "30%", "display": "inline-block", "align-items":"center","justify-content": "center"}),
    html.Br(),
    html.Div([
        html.Label("City"),
        dcc.Dropdown(
            id='city-dropdown',
            options=[
                {'label': 'Warangal', 'value': 4},
                {'label': 'Khammam', 'value': 2},
                {'label': 'Nizamabad', 'value': 3},
                {'label': 'Adilabad', 'value': 0},
                {'label': 'Karimnagar', 'value': 1}
            ],
            value=4
        ),
    html.Br(),
    ], style={"width": "30%", "display": "inline-block", "align-items":"center","justify-content": "center"}),

    dcc.Graph(id="aqi-graph"),
    dcc.Graph(id="temp-graph")
])

# Define the callbacks
@app.callback(
    Output("aqi-graph", "figure"),
    Output("temp-graph", "figure"),
    Input("month-dropdown", "value"),
    Input("city-dropdown", "value")
)
def update_graphs(month, city):
    # Filter data based on month and city
    #aqi_filtered = aqi_data[(aqi_data["Month"] == month) & (aqi_data["City"] == city)]
    #temp_filtered = temp_data[(temp_data["Month"] == month) & (temp_data["City"] == city)
    aqi,heat,city=tango(city)
    aqi_month_data=aqi[aqi['ds'].apply(lambda x:x.month==month and x.year==2023)]
    heat_month_data=heat[heat['ds'].apply(lambda x:x.month==month and x.year==2023)]    
    heat_month_data['temp']=heat_month_data[['yhat','yhat_lower','yhat_upper']].median(axis=1)
    # Create AQI graph
    aqi_fig = px.line(aqi_month_data, x="ds", y="yhat", title="Air Quality Index of {}".format(city.capitalize()),labels=['Date','Air Quality Index'])
    # Create temperature/heatwave graph
    temp_fig = px.scatter(heat_month_data,x='ds', y='temp',title="Heatwave",size="yhat",color="yhat",labels=['Date','Temperature'])
    return aqi_fig, temp_fig
if __name__ == "__main__":
    app.run_server(port=4050)


# In[ ]:




