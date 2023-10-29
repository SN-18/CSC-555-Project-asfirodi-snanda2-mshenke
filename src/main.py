'''

@authors: Firodi, Nanda, Shenker-Tauris
@Github: abhishgain99, SN-18, lorem_ipsum
@file: main.py
@return: returns graph topologies that are location-optimized and plots these optimizations
@see: README.md at https://github.com/SN-18/CSC-555-Project-asfirodi-snanda2-mshenke

#############################################################################################################################################################
'''

# import subprocess

# installs
# def install(package):
#     # subprocess.run(['pip', 'install', '-r', 'requirements.txt']
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])


#Installs
# install("dimod")
# install("pandas")
# install("subprocess")

import numpy as np

if __name__ == "__main__":

    #Imports
    import pandas as pd
    import subprocess
    import sys
    import dimod
    # Provide the path to your Excel file
    file_path = 'your_file.xls'
    import pandas as pd


    file_path = "daily-bike-share.csv"

    # Read the Excel file into a DataFrame

    df = pd.read_csv(file_path)
    print(df.head())


'''
     ###################################################################################################
    What I'm trying to do:
    1. Take the bike-share database
    2. Construct Distance matrix
    3. Use BQM on distance Matrix
    4. By iterating over each node, and it's neighbour
    5. Find the highest energy, report most optimal row(s)
    6. Repeat over several runs by eliminating visited rows, to know resource allocation in descending order
     #################################################################################################################
     
     
     
                                            DISTANCE MATRIX CALCULATIONS CODE:
    #################################################################################################################
    
'''

import pandas
import math
import dimod

#Calculate Euclidean Distance between two bike-sharing nodes
def euclidean(node1,node2):
	d = 0
	for i in range(len(node1)):
		num1 = node1[i]
		num2 = node2[i]
		d = d + math.sqrt(abs((num1)**2 - (num2)**2))
	return d

# Reading the CSV file, splitting into even columns
df = pandas.read_csv('daily-bike-share.csv')
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




#CREATING DISTANCE MATRIX
i=j=0
for index, row in first_frame.iterrows():
	for index2,row2 in second_frame.iterrows():
		if j>=l or i>=l:
			break
		distance_matrix[i][j] = euclidean(row, row2)
		print(distance_matrix[i][j])
		j = j + 1
	i = i + 1
	if i>=l:
		break

#Debug
    # print(row['c1'], row['c2'])
# for i in range(1,len(first_frame)):
# 	for j in range(1,len(second_frame)):
# 		print(first_frame[i],second_frame[j])
# 		distance_matrix.append[i][j] = euclidean(first_frame[i],second_frame[j])
#End of Debug

print("Distance Matrix is: ")
print(distance_matrix)


#Now trying out bqm on distance matrix
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

for node, in range(len(distance_matrix)):
    for neighbor in range(node):
        # node_dict.items():
	# Define a variable for the node
        v_node = f'node_{node}'
        bqm.add_variable(v_node, 1.0)

	    # for neighbor in neighbors:
		# Define a variable for the neighbor
		v_neighbor = f'node_{neighbor}'

		# Assuming you want to add an interaction based on the difference between node and neighbor
		# Adjust this interaction based on your specific problem
		interaction = (node - neighbor) // node

		if interaction != 0:
		    try:
				bqm.add_interaction(v_node, v_neighbor, interaction)
			except:
				pass





'''                                               Distance matrix calculations code end
################################################################################################################################################################
'''





    #debugging for a different xlrd engine
    # if file.endswith('.xlsx'):
    #     if '~$' in file:
    #         pass
    #     else:
    #         file = file.append(file + '/' + file)
    # df = pd.read_excel(file)
    # print("Executing main.py")
    #end of debug



    graph1 ={}
    for i in range(len(df["day"])):
        day = df["day"][i]
        rentals = df["rentals"][i]

        # Initialize an empty list for each day/node
        if day not in graph1:
            graph1[day] = []

        # Add an edge to represent the number of rentals
        for j in range(i + 1, len(df["day"])):
            other_day = df["day"][j]
            other_rentals = df["rentals"][j]

            # print("other_day is: ",other_day)
            # Can define your own criteria for creating edges.
            # Here, we'll create an edge if the absolute difference in rentals is less than or equal to 10.

            if abs(rentals - other_rentals) <= 10:
                graph1[day].append(other_day)

                if other_day not in graph1:
                    graph1[other_day] = []
                    graph1[other_day].append(day)

            #Debug
            # print("graph so far")
            # print(graph1)
            # Print the undirected graph as a dictionary
            #End Debug

        print(graph1)

    #Debug
    #Older Sample Graph
    # Example graph with coordinates and edge weights
    #pass the new graph instead of this one
    # graph = {
    #     (0, 0): {
    #         (1, 0): 1,
    #         (0, 1): 2,
    #     },
    #     (1, 0): {
    #         (0, 0): 1,
    #         (1, 1): 2,
    #     },
    #     (0, 1): {
    #         (0, 0): 2,
    #         (1, 1): 1,
    #     },
    #     (1, 1): {
    #         (1, 0): 2,
    #         (0, 1): 1,
    #     },
    # }
    # bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    #End of Debug
    # quadratic = (np.arange(0, len(df.day)), np.arange(1, len(df.day)), -np.ones(len(df.day)))
    # bqm = dimod.BQM.from_numpy_vectors(len(df.day)*len(df.day), quadratic, 0, "BINARY")

    bqm = dimod.BinaryQuadraticModel(len(df.day)*len(df.day),'BINARY')

    for node, neighbors in graph1.items():
        # Define a variable for the node
        v_node = f'node_{node}'
        bqm.add_variable(v_node, 1.0)

        for neighbor in neighbors:
            # Define a variable for the neighbor
            v_neighbor = f'node_{neighbor}'

            # Assuming you want to add an interaction based on the difference between node and neighbor
            # Adjust this interaction based on your specific problem
            interaction = (node - neighbor)//node

            if interaction!=0:
                try:
                    bqm.add_interaction(v_node, v_neighbor,interaction)
                except:
                    pass


#add an edge between day and other day, if one exists, skip

    for u, neighbors in graph1.items():
        # print("neighbors are: ",neighbors)

        for i in range(len(neighbors)):
            for j in range(i+1,len(neighbors)):
                bqm.add_interaction(i,j,i-j)

        #Debug
        # for v, weight in neighbors.items():
        #     if u < v:  # Only add each edge once to avoid double-counting
        #         bqm.add_interaction(u, v, weight)
        #End of Debug


    sampler = dimod.ExactSolver()
    l=len(bqm)
    print("l is:",l)
    num = 5
    n_a=np.array(bqm)
    response = sampler.sample(n_a.reshape(1,-1))


    #Debug
    # for sample, energy in response.data(fields=['sample', 'energy']):
    #     print(sample, energy)


    # bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
    #                                   {(0, 1): .5, (1, 2): 1.5},
    #                                   1.4,
    #                                   dimod.Vartype.SPIN)

    #End of Debug


    print("BQM is: \n")
    print(bqm)
    bqm.viewitems()




















