'''

@authors: Firodi, Nanda, Shenker-Tauris
@Github: abhishgain99, SN-18, lorem_ipsum
@file: main.py
@return: returns graph topologies that are location-optimized and plots these optimizations
@see: README.md at https://github.com/SN-18/CSC-555-Project-asfirodi-snanda2-mshenke

#############################################################################################################################################################
'''


# installs
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


#Installs
install("dimod")
install("pandas")
install("subprocess")


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
                    bqm.add_interaction(v_node, v_neighbor, interaction)
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
    response = sampler.sample(bqm)


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




















