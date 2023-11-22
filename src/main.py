# -*- coding: utf-8 -*-

'''

@authors: Firodi, Nanda, Shenker-Tauris
@Github: abhishgain99, SN-18, lorem_ipsum
@file: main_1.py
@return: returns graph topologies that are location-optimized and plots these optimizations
@see: README.md at https://github.com/SN-18/CSC-555-Project-asfirodi-snanda2-mshenke

#############################################################################################################################################################
'''

import pandas as pd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads
import matplotlib.pyplot as plt
from shapely.geometry import shape
import copy
import math
import dimod
import numpy as np
from dwave.system import LeapHybridSampler

api_token = 'DEV-d14fa731662f7519b2a37fd6643e711d02208cac'

################################
#      DATA EXTRACT STAGE      #
#################################

"""## DAYTIME AND RIDECOUNT"""


station_df = pd.read_csv('JC-202307-citibike-tripdata.csv').dropna()
station_df['start_time'] = pd.to_datetime(station_df['started_at'])
station_df['end_time'] = pd.to_datetime(station_df['ended_at'])

start_time = datetime.strptime('06:00:00', '%H:%M:%S').time()
end_time = datetime.strptime('18:00:00', '%H:%M:%S').time()

station_df['daytime'] = ((station_df['start_time'].dt.time >= start_time) & (station_df['start_time'].dt.time <= end_time)) & ((station_df['end_time'].dt.time >= start_time) & (station_df['end_time'].dt.time <= end_time))
ride_count_by_daytime = station_df.groupby(['end_station_id', 'daytime'])['ride_id'].size().reset_index()

station_df['end_lat'] = (station_df['end_lat'] * 10000000).astype(int)
station_df['end_lat'] /= 10000000
station_df['end_lat'] = station_df['end_lat'].round(6)
station_df['end_lng'] = (station_df['end_lng'] * 10000000).astype(int)
station_df['end_lng'] /= 10000000
station_df['end_lng'] = station_df['end_lng'].round(6)

merged_df = ride_count_by_daytime.merge(df[['end_station_name', 'end_station_id', 'end_lat', 'end_lng']], on='end_station_id', how='left').drop_duplicates().reset_index(drop=True).rename(columns={'ride_id': 'ride_count'})

lat_lng_df = copy.deepcopy(merged_df[['ride_count', 'end_station_name', 'end_station_id', 'end_lat', 'end_lng', 'daytime']])
lat_lng_df = lat_lng_df.drop_duplicates()
lat_lng_df = lat_lng_df[lat_lng_df['end_station_id'].str.contains('JC')==False]
lat_lng_df = lat_lng_df[lat_lng_df['end_station_id'].str.contains('HB')==False]
print(lat_lng_df)

"""# CRIME, POPULAR LOCATIONS AND FOOTFALL DATA

## Load data
"""
citibike_data = lat_lng_df
unique_citibike_stations = citibike_data.groupby(['end_station_id', 'daytime']).first().reset_index()
unique_citibike_stations = unique_citibike_stations[unique_citibike_stations['end_station_id'].str.contains('JC')==False].reset_index()
unique_citibike_stations = unique_citibike_stations[unique_citibike_stations['end_station_id'].str.contains('HB')==False].reset_index()

crime_places = pd.read_csv('Crime_Map_.csv')
start_date = pd.to_datetime('2023-07-01')
end_date = pd.to_datetime('2023-07-31')
crime_places['CMPLNT_FR_DT'] = pd.to_datetime(crime_places['CMPLNT_FR_DT'], errors='coerce', format='%m/%d/%Y')
crime_places_df = crime_places[(crime_places['CMPLNT_FR_DT'] >= start_date) & (crime_places['CMPLNT_FR_DT'] <= end_date) & (crime_places['CMPLNT_FR_DT'].notnull())]

landmark_data = pd.read_csv('Point_Of_Interest.csv')

"""## Convert to GeoDataFrame and Reproject to a suitable CRS for New York"""

unique_citibike_stations_gdf = gpd.GeoDataFrame(unique_citibike_stations, geometry=gpd.points_from_xy(unique_citibike_stations.end_lng, unique_citibike_stations.end_lat))
unique_citibike_stations_gdf.crs = 'epsg:4326'
unique_citibike_stations_gdf = unique_citibike_stations_gdf.to_crs('epsg:2263')

crime_places_gdf = gpd.GeoDataFrame(crime_places_df, geometry=gpd.points_from_xy(crime_places_df.Longitude, crime_places_df.Latitude))
crime_places_gdf.crs = 'epsg:4326'
crime_places_gdf = crime_places_gdf.to_crs('epsg:2263')

landmark_gdf = gpd.GeoDataFrame(landmark_data)
landmark_gdf['geometry'] = landmark_gdf['the_geom'].apply(lambda x: shape(loads(x)))
landmark_gdf.crs = 'epsg:4326'
landmark_gdf = landmark_gdf.to_crs('epsg:2263')

"""## Buffer creation around each Citibike station"""

crime_buffer_distance_feet = 5000 * 3.28084
unique_citibike_stations_gdf['crime_buffer'] = unique_citibike_stations_gdf.geometry.buffer(crime_buffer_distance_feet)

landmark_buffer_distance_feet = 500 * 3.28084
unique_citibike_stations_gdf['landmark_buffer'] = unique_citibike_stations_gdf.geometry.buffer(landmark_buffer_distance_feet)

"""## Create spatial index for landmarks and crimes"""

crime_spatial_index = landmark_gdf.sindex
landmark_spatial_index = landmark_gdf.sindex

"""## Function to count landmarks and crimes within buffer for each station"""

def count_nearby_crimes(station):
    possible_crimes_index = list(landmark_spatial_index.intersection(station['crime_buffer'].bounds))
    possible_crimes = landmark_gdf.iloc[possible_crimes_index]
    precise_crimes = possible_crimes[possible_crimes.intersects(station['crime_buffer'])]
    return len(precise_crimes)

def count_nearby_landmarks(station):
    possible_landmarks_index = list(landmark_spatial_index.intersection(station['landmark_buffer'].bounds))
    possible_landmarks = landmark_gdf.iloc[possible_landmarks_index]
    precise_landmarks = possible_landmarks[possible_landmarks.intersects(station['landmark_buffer'])]
    return len(precise_landmarks)

"""## Apply function to count landmarks for each station"""

unique_citibike_stations_gdf['crime_count'] = unique_citibike_stations_gdf.apply(count_nearby_crimes, axis=1)
unique_citibike_stations_gdf['landmark_count'] = unique_citibike_stations_gdf.apply(count_nearby_landmarks, axis=1)
print(unique_citibike_stations_gdf[['end_station_id', 'crime_count', 'landmark_count']])

"""## Footfall Count"""

start_station_footfall = station_df.groupby('start_station_id').size()
end_station_footfall = station_df.groupby('end_station_id').size()
total_station_footfall = start_station_footfall.add(end_station_footfall, fill_value=0)
footfall_df = pd.DataFrame(total_station_footfall, columns=['footfall_count']).rename_axis('end_station_id').reset_index()

citibike_data_df_final = citibike_data.merge(unique_citibike_stations_gdf[['end_station_id', 'crime_count', 'landmark_count']], how='left', on='end_station_id')
citibike_data_df_final = citibike_data_df_final.merge(footfall_df[['end_station_id', 'footfall_count']], how='left', on='end_station_id')
print(citibike_data_df_final)

"""## Deepcopy for BQM"""

df = copy.deepcopy(citibike_data_df_final[['ride_count', 'daytime', 'crime_count', 'landmark_count', 'footfall_count']])
df = df.drop_duplicates()
print(df)



################################
###### BQM Calculations #########
#################################




def BQM(df):
    def euclidean(node1,node2):
        d = 0
        for i in range(len(node1)):
            num1 = int(node1[i])
            num2 = int(node2[i])
            d = d + math.sqrt(abs((num1)**2 - (num2)**2))
        return d

    l = len(df)
    first_frame = df[:l//2]
    second_frame = df[l//2:]

    ## Construct a (n/2)*(n/2) distance matrix full of zeros
    distance_matrix = []
    for i in range(l//2):
        small_arr = []
        for j in range(l//2):
            small_arr.append(0)
        distance_matrix.append(small_arr)

    i=j=0

    ## CREATING DISTANCE MATRIX
    for index, row in first_frame.iterrows():
        for index2,row2 in second_frame.iterrows():
            if j>=l//2 or i>=l//2:
                break
            distance_matrix[i][j] = euclidean(row, row2)
            j = j + 1
        i = i + 1
        if i>=l:
            break

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    node_dict = []
    for index, row in df.iterrows():
        node = []
        for element in row:
            node.append(element)
        node_dict.append(node)

    distance_dict = [0]
    for i in range(2,len(node_dict)):
        distance = euclidean(node_dict[i],node_dict[0])
        distance_dict.append(distance)

    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            i + len(distance_matrix[i])
            node = i
            neighbor = i + j
            v_node = f'node_{node}'
            bqm.add_variable(v_node, 1.0)
            v_neighbor = f'node_{neighbor}'
            bqm.add_variable(v_neighbor, 1.0)
            if node == 0:
                div = 1
            else:
                div = node
            interaction = (node - neighbor) // div + distance_matrix[i][j]
            if interaction != 0:
                try:
                    bqm.add_interaction(v_node, v_neighbor, interaction)
                except:
                    pass

    points_to_keep = list(range(l // 2))

    energy_store = list()
    while len(points_to_keep) >= 10:
        # Create a BQM
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        for i in range(len(points_to_keep)):
            for j in range(len(points_to_keep)):
                if i == j:
                    continue
                v_node = f'node_{points_to_keep[i]}'
                v_neighbor = f'node_{points_to_keep[j]}'
                interaction = (points_to_keep[i] - points_to_keep[j]) // (points_to_keep[i] + 1) + distance_matrix[points_to_keep[i]][points_to_keep[j]]
                if interaction != 0:
                    bqm.add_variable(v_node, 1.0)
                    bqm.add_variable(v_neighbor, 1.0)
                    bqm.add_interaction(v_node, v_neighbor, interaction)

        # Use the Leap Hybrid Sampler
        sampler = LeapHybridSampler(token=api_token)
        response = sampler.sample(bqm)

        # Get the sample with maximum energy
        max_energy_sample = next(response.samples())
        max_energy = response.first.energy

        # Remove the point with maximum energy
        max_energy_point = list(max_energy_sample.keys())[0]
        point_to_remove = int(max_energy_point.split('_')[1])
        points_to_keep.remove(point_to_remove)

        print("Sample with Maximum Energy:", max_energy_sample)
        print("Energy:", max_energy)
        energy_store.append([point_to_remove, max_energy])

    ## Final output
    final_energies = {}
    for i in range(len(energy_store)-1):
        final_energies[energy_store[i][0]] = abs(energy_store[i][1]-energy_store[i+1][1])

    top_contenders = sorted(final_energies.items(), key=lambda x:x[1], reverse=True)
    print("***** Top contenders *****\n", top_contenders)

    print("# Remaining Points:", points_to_keep)



"""## Shuffling data and calling BQM for all"""

sample1 = copy.deepcopy(df)
BQM(sample1)

sample2 = copy.deepcopy(df.sample(frac=1))
BQM(sample2)

sample3 = copy.deepcopy(df.sample(frac=1))
BQM(sample3)

sample4 = copy.deepcopy(df.sample(frac=1))
BQM(sample4)

