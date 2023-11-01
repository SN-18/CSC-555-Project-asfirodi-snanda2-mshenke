
'''

@authors: Firodi, Nanda, Shenker-Tauris
@Github: abhishgain99, SN-18, lorem_ipsum
@file: main_1.py
@return: returns graph topologies that are location-optimized and plots these optimizations
@see: README.md at https://github.com/SN-18/CSC-555-Project-asfirodi-snanda2-mshenke

#############################################################################################################################################################
'''




import pandas
import math
import dimod
import numpy as np
from dwave.system import LeapHybridSampler

api_token = 'DEV-1a546ca54c56c06669335e81b9e804aaae544f1d'


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





def euclidean(node1,node2):
	d = 0
	for i in range(len(node1)):
		num1 = node1[i]
		num2 = node2[i]
		d = d + math.sqrt(abs((num1)**2 - (num2)**2))
	return d

# reading the CSV file, splitting into even columns
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


i=j=0

#CREATING DISTANCE MATRIX
for index, row in first_frame.iterrows():

	#Debug
	# print("I am inside this for loop: ")
	# print(row)
	# print(index)
	# print(row[0])
	# print(type(row))
	#End of Debug

	for index2,row2 in second_frame.iterrows():
		if j>=l//2 or i>=l//2:
			break
		distance_matrix[i][j] = euclidean(row, row2)
		print(distance_matrix[i][j])
		j = j + 1


	i = i + 1
	if i>=l:
		break

    # print(row['c1'], row['c2'])





# for i in range(1,len(first_frame)):
# 	for j in range(1,len(second_frame)):
# 		print(first_frame[i],second_frame[j])
# 		distance_matrix.append[i][j] = euclidean(first_frame[i],second_frame[j])

print("Distance Matrix is: ")
print(distance_matrix)
# displaying the contents of the CSV file
# print(csvFile)

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

# sampler = dimod.ExactSolver()
# l=len(bqm)
# print("l is:",l)
# num = 3
# n_a=np.array(bqm)
# response = sampler.sample(n_a.reshape(3,-1))

# sampler = dimod.ExactSolver()
# response = sampler.sample(bqm)
#
# # Print the results
# for sample, energy in response.data(fields=['sample', 'energy']):
#     print(sample, energy)
points_to_keep = list(range(l // 2))
# while len(points_to_keep) >= 10:
#     # Create a BQM
#     bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
#
#     for i in range(len(points_to_keep)):
#         for j in range(len(points_to_keep)):
#             if i == j:
#                 continue
#             v_node = f'node_{points_to_keep[i]}'
#             v_neighbor = f'node_{points_to_keep[j]}'
#             interaction = (points_to_keep[i] - points_to_keep[j]) // (points_to_keep[i] + 1) + distance_matrix[points_to_keep[i]][points_to_keep[j]]
#             if interaction != 0:
#                 bqm.add_variable(v_node, 1.0)
#                 bqm.add_variable(v_neighbor, 1.0)
#                 bqm.add_interaction(v_node, v_neighbor, interaction)
#
#     # Solve the BQM to find the sample with maximum energy
#     sampler = dimod.ExactSolver()
#     response = sampler.sample(bqm, num_reads=1)  # Set num_reads=1 to find a single maximum energy sample
#
#     # Get the sample with maximum energy
#     max_energy_sample = next(response.samples())
#     max_energy = response.first.energy
#
#     # Remove the point with maximum energy
#     max_energy_point = max_energy_sample.keys()
#     points_to_keep.remove(int(max_energy_point[0].split('_')[1]))
#
#     print("Sample with Maximum Energy:")
#     print(max_energy_sample)
#     print("Energy:", max_energy)
#
# print("Remaining Points:")
# print(points_to_keep)

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

print("Remaining Points:")
print(points_to_keep)


# sampler = LeapHybridSampler(token=api_token)  # Pass the API token here
# response = sampler.sample(bqm)

# Print the results
# for sample, energy in response.data(fields=['sample', 'energy']):
#     print(sample, energy)
#
# max_energy_sample = next(response.samples())
# max_energy = response.first.energy
#
# print("Sample with Maximum Energy:")
# print(max_energy_sample)
# print("Energy:", max_energy)