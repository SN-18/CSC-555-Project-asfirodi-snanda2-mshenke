'''

@authors: Firodi, Nanda, Shenker-Tauris
@Github: abhishgain99, SN-18, lorem_ipsum
@file: main.py
@return: returns graph topologies that are location-optimized and plots these optimizations
@see: README.md at https://github.com/SN-18/CSC-555-Project-asfirodi-snanda2-mshenke

#############################################################################################################################################################
'''


import argparse
import dimod
import sys
import networkx as nx
import numpy as np
import matplotlib as plt




def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("dimod")


if "__name__" == "main":

    import dimod
    import subprocess
    import sys
    

    # Example graph with coordinates and edge weights
    graph = {
        (0, 0): {
            (1, 0): 1,
            (0, 1): 2,
        },
        (1, 0): {
            (0, 0): 1,
            (1, 1): 2,
        },
        (0, 1): {
            (0, 0): 2,
            (1, 1): 1,
        },
        (1, 1): {
            (1, 0): 2,
            (0, 1): 1,
        },
    }
    
    
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            if u < v:  # Only add each edge once to avoid double-counting
                bqm.add_interaction(u, v, weight)
    
    
    
    sampler = dimod.ExactSolver()
    response = sampler.sample(bqm)
    
    for sample, energy in response.data(fields=['sample', 'energy']):
        print(sample, energy)
    
  
    
    print(bqm)
  

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("dimod")



if "__name__" == "main":

    import dimod
    import subprocess
    import sys
    

    # Example graph with coordinates and edge weights
    graph = {
        (0, 0): {
            (1, 0): 1,
            (0, 1): 2,
        },
        (1, 0): {
            (0, 0): 1,
            (1, 1): 2,
        },
        (0, 1): {
            (0, 0): 2,
            (1, 1): 1,
        },
        (1, 1): {
            (1, 0): 2,
            (0, 1): 1,
        },
    }
    
    
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            if u < v:  # Only add each edge once to avoid double-counting
                bqm.add_interaction(u, v, weight)
    
    
    
    sampler = dimod.ExactSolver()
    response = sampler.sample(bqm)
    
    for sample, energy in response.data(fields=['sample', 'energy']):
        print(sample, energy)
    
    print(bqm)

#debug
    # bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
    #                                   {(0, 1): .5, (1, 2): 1.5},
    #                                   1.4,
    #                                   dimod.Vartype.SPIN)
    
    
    
    # bqm.viewitems()
    
    
    
    
    
    
    
    
    
    
# bqm.viewitems()
    
    
    
    
    
    
    
    
    
    


