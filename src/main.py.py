
##########################
#        IMPORTS         #
##########################


from datetime import datetime
import copy
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.wkt import loads
import matplotlib.pyplot as plt
from shapely.geometry import shape
import math
import dimod
import numpy as np
from dwave.system import LeapHybridSampler

api_token = 'DEV-1a546ca54c56c06669335e81b9e804aaae544f1d'



"""Idea: Find the range of locations in terms of latitude and longitude. Then in this range, define some threshold and find all the points in between."""


################################
#      DATA EXTRACT STAGE      #
#################################

df = pd.read_csv('JC-202307-citibike-tripdata.csv').dropna()
df['start_time'] = pd.to_datetime(df['started_at'])
df['end_time'] = pd.to_datetime(df['ended_at'])

start_time = datetime.strptime('06:00:00', '%H:%M:%S').time()
end_time = datetime.strptime('18:00:00', '%H:%M:%S').time()

df['daytime'] = ((df['start_time'].dt.time >= start_time) & (df['start_time'].dt.time <= end_time)) & ((df['end_time'].dt.time >= start_time) & (df['end_time'].dt.time <= end_time))
ride_count_by_daytime = df.groupby(['end_station_id', 'daytime'])['ride_id'].size().reset_index()

df['end_lat'] = (df['end_lat'] * 10000000).astype(int)
df['end_lat'] /= 10000000
df['end_lat'] = df['end_lat'].round(6)
df['end_lng'] = (df['end_lng'] * 10000000).astype(int)
df['end_lng'] /= 10000000
df['end_lng'] = df['end_lng'].round(6)

merged_df = ride_count_by_daytime.merge(df[['end_station_name', 'end_station_id', 'end_lat', 'end_lng']], on='end_station_id', how='left').drop_duplicates().reset_index(drop=True).rename(columns={'ride_id': 'ride_count'})

lat_lng_df = copy.deepcopy(merged_df[['ride_count', 'end_station_name', 'end_station_id', 'end_lat', 'end_lng', 'daytime']])
lat_lng_df = lat_lng_df.drop_duplicates()
print(lat_lng_df)



#########################
#  DATA LOAD STAGE   #
##########################

file = open('distance.txt', 'w')
for d in distances:
  file.write(str(d)+"\n")

from google.colab import files
files.download('distance.txt')



with open('distance.txt') as file:
  distances = file.read().splitlines()

# Correct one with distances.tst present
import pandas as pd

bus_stop_df = lat_lng_df
crime_places = pd.read_csv('Crime_Map_.csv')
start_date = pd.to_datetime('2023-07-01')
end_date = pd.to_datetime('2023-07-31')
crime_places['CMPLNT_FR_DT'] = pd.to_datetime(crime_places['CMPLNT_FR_DT'], errors='coerce', format='%m/%d/%Y')
crime_places_df = crime_places[(crime_places['CMPLNT_FR_DT'] >= start_date) & (crime_places['CMPLNT_FR_DT'] <= end_date) & (crime_places['CMPLNT_FR_DT'].notnull())]

threshold_distance = 5000

i = 0
print("distances", len(distances))
for index, bus_stop in bus_stop_df.iterrows():
  crime_count = 0
  for d in range(i, i+len(crime_places_df)):
    print(index, i)
    if i == len(distances) - 1:
      break
    if float(distances[d]) <= threshold_distance:
      crime_count -= 1
    bus_stop_df.at[index, 'nearby_crime_count'] = crime_count
    i = d

print(bus_stop_df[['ride_count', 'end_station_name', 'end_station_id', 'end_lat', 'end_lng', 'daytime', 'nearby_crime_count']])

"""Crime Data count"""


landmark_data = pd.read_csv("Point_Of_Interest.csv")

citibike_data = pd.read_csv("JC-202307-citibike-tripdata.csv")

# Grouping by end_station_id and keep the first occurrence
unique_citibike_stations = citibike_data.groupby('end_station_id').first().reset_index()

unique_citibike_stations

# Converting to landmark data to GeoDataFrame with geometry
landmark_gdf = gpd.GeoDataFrame(landmark_data)
landmark_gdf['geometry'] = landmark_gdf['the_geom'].apply(lambda x: shape(loads(x)))
landmark_gdf.crs = 'epsg:4326'

# Group by end_station_id and keep the first occurrence for citibike data
unique_citibike_stations = citibike_data.groupby('end_station_id').first().reset_index()

# Convert unique Citibike stations to GeoDataFrame
unique_citibike_stations_gdf = gpd.GeoDataFrame(unique_citibike_stations,
                                                geometry=gpd.points_from_xy(unique_citibike_stations.end_lng, unique_citibike_stations.end_lat))
unique_citibike_stations_gdf.crs = 'epsg:4326'

# Reproject to a suitable CRS for New York
landmark_gdf = landmark_gdf.to_crs('epsg:2263')
unique_citibike_stations_gdf = unique_citibike_stations_gdf.to_crs('epsg:2263')

# Buffer creation around each Citibike station
buffer_distance_feet = 500 * 3.28084  # Convert 500 meters to feet
unique_citibike_stations_gdf['buffer'] = unique_citibike_stations_gdf.geometry.buffer(buffer_distance_feet)

# Create spatial index for landmarks
spatial_index = landmark_gdf.sindex

# Function to count landmarks within buffer for each station
def count_nearby_landmarks(station):
    possible_matches_index = list(spatial_index.intersection(station['buffer'].bounds))
    possible_matches = landmark_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(station['buffer'])]
    return len(precise_matches)

# Apply function to count landmarks for each station
unique_citibike_stations_gdf['landmark_count'] = unique_citibike_stations_gdf.apply(count_nearby_landmarks, axis=1)

# Result
print(unique_citibike_stations_gdf[['end_station_id', 'landmark_count']])

bus_stop_df_final = bus_stop_df.merge(unique_citibike_stations_gdf[['end_station_id', 'landmark_count']], how="left", on="end_station_id")
bus_stop_df_final = bus_stop_df_final.fillna(0)
print("**bus_stop_df_final", bus_stop_df_final)



df = copy.deepcopy(bus_stop_df_final[['ride_count', 'nearby_crime_count', 'landmark_count']])
df = df.drop_duplicates()
print(df)






###################################
# DISTANCE MATRIX CALCULATIONS CODE #
###################################

def euclidean(node1,node2):
	d = 0
	for i in range(len(node1)):
		num1 = int(node1[i])
		num2 = int(node2[i])
		d = d + math.sqrt(abs((num1)**2 - (num2)**2))
	return d

# reading the CSV file, splitting into even columns
# df = pandas.read_csv('daily-bike-share.csv')
print(df.head())
l = len(df)
first_frame = df[:l//2]
second_frame = df[l//2:]

#Construct a (n/2)*(n/2) distance matrix full of zeros
distance_matrix = []
for i in range(l//2):
	small_arr = []
	for j in range(l//2):
		small_arr.append(0)
	distance_matrix.append(small_arr)


i=j=0

#CREATING DISTANCE MATRIX
for index, row in first_frame.iterrows():
	for index2,row2 in second_frame.iterrows():
		if j>=l//2 or i>=l//2:
			break
		distance_matrix[i][j] = euclidean(row, row2)
		print(distance_matrix[i][j])
		j = j + 1
	i = i + 1
	if i>=l:
		break


print("Distance Matrix is: ")
print(distance_matrix)


###############################################
#             BQM Calculations                #
###############################################




bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

node_dict = []
for index, row in df.iterrows():
	node = []
	for element in row:
		node.append(element)
	node_dict.append(node)

print(node_dict)


distance_dict = [0]
for i in range(2,len(node_dict)):
	print(node_dict[i])
	distance = euclidean(node_dict[i],node_dict[0])
	distance_dict.append(distance)

print(distance_dict)


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

    print("Sample with Maximum Energy:")
    print(max_energy_sample)
    print("Energy:", max_energy)
    energy_store.append([point_to_remove, max_energy])

final_energies = {}
for i in range(len(energy_store)-1):
  final_energies[energy_store[i][0]] = abs(energy_store[i][1]-energy_store[i+1][1])
print(final_energies)
print("top contendars", sorted(final_energies.items(), key=lambda x:x[1]))
  

print("Remaining Points:")
print(points_to_keep)

