import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time
import datetime
import zarr
import time as tslp
import os
import webbrowser
from explainabletf_utils.models import DGCN
from explainabletf_utils.visualize import *
from explainabletf_utils.utils import *
import pickle
from streamlit_folium import st_folium

torch.set_num_threads(4)

PATH = './data/dataset/'
ref = [37,11,11,16,25,11,34,10,38]

device = torch.device('cpu')

para = {}
para['time_invertal'] = 5
para['horizon'] = 15
para['observation'] = 20
para['nb_node'] = 193
para['dim_feature'] = 128

min_date = date(2018, 1, 1)
max_date = date(2022, 12, 15)
step = datetime.timedelta(minutes=2)

st.set_page_config(layout="wide", page_title="Explainable Traffic Forecasting|DiTTlab", page_icon=":blue_car:")

tab0, tab1, tab2, tab3 = st.tabs(["Highways", "Prediction", "Interpretation", "Demand"])

########################## LOAD MODEL ##################################
@st.cache_data
def get_directed_connection():
    Av = pure_adj(3)
    Aq = pure_adj(9)

    return Av, Aq


@st.cache_data
def get_roads():
    with open('./dataset/segments.pkl', 'rb') as f:
        roads = pickle.load(f)

    lengths = [len(s[0]) for s in list(roads.values())]
    indicators = np.zeros(sum(lengths))
    start = 0
    for i in range(len(roads)):
        indicators[start:start+lengths[i]] = i+1
        start = start+lengths[i]

    return roads, lengths, indicators

@st.cache_resource
def load_predictor(para):
    para = {}
    para['time_invertal'] = 5
    para['horizon'] = 15
    para['observation'] = 20
    para['nb_node'] = 193
    para['dim_feature'] = 128
    A = adjacency_matrix(3)
    B = adjacency_matrix(9)

    model = DGCN(para, A, B, return_interpret=True, uncertainty=False)
    model.load_state_dict(torch.load('./dataset/pretrained/predictor.pt', map_location=device))
    return model

def load_data(para, year, day, period, time):
    T = para['horizon']# + para['observation']
    dt = zarr.open('./dataset/'+year+'.zarr', mode='r')
    if period == 'Morning':
        V = np.transpose(dt.speed_morning[day,:,time:time+20], (1,0))
        Q = np.transpose(dt.flow_morning[day,:,time:time+20], (1,0))
    if period == 'Afternoon/Evening':
        V = np.transpose(dt.speed_evening[day,:,time:time+20], (1,0))
        Q = np.transpose(dt.flow_evening[day,:,time:time+20], (1,0))

    V[V>130] = 100.
    V = V/130.

    Q[Q>3000] = 1000.
    Q = Q/3000.
    x = np.stack([V,Q], -1)
    return torch.Tensor(x).unsqueeze(0).float()


########################## STREAMLIT ##################################
#st.sidebar.image("./UQnet_utils/logo/dittlab.png", width=200)
st.sidebar.header('Explainable AI-based traffic forecasting')
st.sidebar.write("by Guopeng LI, Victor Knoop, and Hans van Lint")

model = load_predictor(para)
links, lengths, indicators = get_roads()
Av, Aq = get_directed_connection()
linestrings = create_linestrings(links, lengths)

selected_date = st.sidebar.date_input("Select a date", date(2018, 3, 1), min_value=min_date, max_value=max_date)

year_i = str(selected_date.year)
day_i = int(selected_date.strftime('%j'))

period = st.sidebar.selectbox('Choose a period', ['Morning', 'Afternoon/Evening'])

if period == 'Morning':
    timeseries = []
    start = time(7, 40)
    end = time(11, 0)
    while start <= end:
        timeseries.append(start.strftime('%H:%M'))
        start = (datetime.datetime.combine(datetime.date(1, 1, 1), start) + step).time()
    selected_time = st.sidebar.selectbox("Select a time", timeseries)
    ti = timeseries.index(selected_time)

if period == 'Afternoon/Evening':
    timeseries = []
    start = time(14, 40)
    end = time(21, 0)
    while start <= end:
        timeseries.append(start.strftime('%H:%M'))
        start = (datetime.datetime.combine(datetime.date(1, 1, 1), start) + step).time()
    selected_time = st.sidebar.selectbox("Select a time", timeseries)
    ti = timeseries.index(selected_time)

prediction = st.sidebar.checkbox('Show prediction')
interpret = st.sidebar.checkbox('Show model interpretation')
demand_estimate = st.sidebar.checkbox('Show demand estimation')

with tab0:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Highway network around Amsterdam")
        st.image("./dataset/imgs/maps.png", width=500, output_format='PNG')
    with col2:
        st.subheader("Dirving directions and road numbers")
        st.image("./dataset/imgs/vectormaps.png", width=450, output_format='PNG')

if prediction or interpret or demand_estimate:
    x = load_data(para, year_i, day_i, period, ti)
    y, m1, m2, demand = model(x, 0.)

    xin = x[0].detach().numpy()
    y = y[0].detach().numpy()
    m1 = m1.detach().numpy()
    m2 = m2.detach().numpy()
    demand = np.mean(demand[0].detach().numpy(), 0)

if prediction:
    with tab1:
        quantity = st.selectbox('Choose the variable to predict', ['speed', 'flow'])
        if st.button('START PREDICTION!'):
            col1, col2 = st.columns([6,6])

            with col1:
                p1 = st.empty()
                for sr in range(1,16):
                    if quantity == 'speed':
                        fig1, fig2 = dynamic_speed(links, xin[...,0]*130, y[...,0]*130, step=sr)
                    if quantity == 'flow':
                        fig1, fig2 = dynamic_flow(links, xin[...,1]*3000, y[...,1]*3000, step=sr)

                    p1.write(fig1)
                    tslp.sleep(0.3)
            
            with col2:
                p2 = st.empty()
                for sr in range(1,16):
                    if quantity == 'speed':
                        fig1, fig2 = dynamic_speed(links, xin[...,0]*130, y[...,0]*130, step=sr)
                    if quantity == 'flow':
                        fig1, fig2 = dynamic_flow(links, xin[...,1]*3000, y[...,1]*3000, step=sr)

                    p2.write(fig2)


                # for sr in range(1,16):
                #     if quantity == 'speed':
                #         fig1, fig2 = dynamic_speed(links, xin[...,0]*130, y[...,0]*130, step=sr)
                #     if quantity == 'flow':
                #         fig1, fig2 = dynamic_flow(links, xin[...,1]*3000, y[...,1]*3000, step=sr)

                #     with p1.container:
                #         col1, col2 = st.columns([6,6])
                #         with col1:
                #             st.pyplot(fig1, clear_figure=True)
                    #col2.pyplot(fig2, clear_figure=True)

                # with col1:
                #     p1.write(fig1)
                #     #st.pyplot(fig1, clear_figure=True)
                # with col2:
                #     p2.write(fig2)
                #     #st.pyplot(fig2, clear_figure=True)

                
                    #st.pyplot(fig2, clear_figure=True)



if interpret:
    with tab2:
        col1, col2 = st.columns([6,5])
        with col1:
            step = st.slider('Choose prediction horizon (min)', 2, 30, step=2)
        with col2:
            variable = st.selectbox('Choose a variable', ['flow', 'speed'])
        hi = int(step/2)-1
        if variable == 'speed':
            map_v = draw_status_folium(y[hi,:,0]*130, links, lengths, linestrings, zoom_start=11, mode='speed')
            with col1:
                st.write('Speed:')
                st_mapv = st_folium(map_v, width=500, height=400)
                if st_mapv['last_active_drawing']:
                    sensor_id = st_mapv['last_active_drawing']['properties']['ids']
                    sensor_id = int(indicators[sensor_id])-1

                    with col2:
                        st.markdown('Spatial influence at Link id:' + str(sensor_id+1))
                        fig = draw_interpretation(links, Av, m1[hi], sensor_id, 'speed')
                        st.pyplot(fig, clear_figure=True)
        
        if variable == 'flow':
            map_v = draw_status_folium(y[hi,:,1]*3000, links, lengths, linestrings, zoom_start=11, mode='flow')
            with col1:
                st.write('Flow:')
                st_mapv = st_folium(map_v, width=500, height=400)
                if st_mapv['last_active_drawing']:
                    sensor_id = st_mapv['last_active_drawing']['properties']['ids']
                    sensor_id = int(indicators[sensor_id])-1

                    with col2:
                        st.markdown('Spatial influence at Link id:' + str(sensor_id+1))
                        fig = draw_interpretation(links, Aq, m2[hi], sensor_id)
                        st.pyplot(fig, clear_figure=True)
            

if demand_estimate:
    with tab3:
        col1, col2 = st.columns([8,2])
        map_demand = draw_demand(demand, links, lengths, linestrings, zoom_start=11)
        with col1:
            st.subheader('Demand estimation')
            st_map = st_folium(map_demand, width=700, height=450)
            if st_map['last_active_drawing']:
                sensor_id = st_map['last_active_drawing']['properties']['ids']
                sensor_id = indicators[sensor_id]
                demand_show = round(st_map['last_active_drawing']['properties']['demand'],5)

                with col2:
                    st.markdown('Link id:')
                    st.markdown(sensor_id)
                    st.markdown('Estimated demand (veh/lane/hour):')
                    st.markdown(demand_show)
