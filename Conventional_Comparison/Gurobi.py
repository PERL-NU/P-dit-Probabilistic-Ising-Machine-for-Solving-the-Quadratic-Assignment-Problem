import sys
import os
import subprocess
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

data_path = "qapdata"
solution_path = "qapsoln"

file_names = []

try:
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isfile(item_path):
            file_names.append(item)
except FileNotFoundError:
    print(f"Error: Folder not found at '{data_path}'")
except Exception as e:
    print(f"An error occurred: {e}")

print(len(file_names))
for name in file_names:
    bestSolution = -1
    try:
        with open(solution_path+"/"+name[:-3]+"sln", 'r') as file:
            for line in file:
                parts = line.strip().split() 
                bestSolution = int(parts[1])
                break
    except Exception:
        continue


    fileName = data_path+"/"+name
    print(fileName)
    readFile = open(fileName, "r")
    data = readFile.read().split()

    if os.path.exists(name+"_1000_log_bestSolStop.txt"):
        continue

    print(data)

    N = int(data[0])

    data = data[1:]

    D = np.zeros((N,N))
    F = np.zeros((N,N))

    for r in range(N):
        for c in range(N):
            D[r,c] = int(data[r*N+c])
            F[r,c] = int(data[N*N+r*N+c])
    n = N

    non_zero_F = []
    for i in range(n):
        for j in range(n):
            if F[i,j] != 0:
                non_zero_F.append({'i': i, 'j': j, 'val': F[i,j]})

    non_zero_D = []
    for k in range(n):
        for l in range(n):
            if D[k,l] != 0:
                non_zero_D.append({'k': k, 'l': l, 'val': D[k,l]})


    model = gp.Model("QAP")
    model.setParam('TimeLimit', 1000)
    model.Params.LogFile = name+"_1000_log_bestSolStop.txt"
    model.Params.Threads = 20
    model.params.SoftMemLimit = 40 # GB  
    model.setParam("NodefileStart", 40.0)
    
    model.params.BestObjStop = bestSolution + 0.001

    # Variables: x[i, k] = 1 if facility i is assigned to location k
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Constraints:
    # Each facility i is assigned to exactly one location k
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(n)) == 1, name=f"assign_facility_{i}")

    # Each location k is assigned exactly one facility i
    for k in range(n):
        model.addConstr(gp.quicksum(x[i, k] for i in range(n)) == 1, name=f"assign_location_{k}")

    # Objective function
    obj = gp.QuadExpr()

    # --- Efficiently build quadratic terms ---
    q_coeffs = []
    q_vars1 = []
    q_vars2 = []

    for f_term in non_zero_F: # Iterate M_F times
        i = f_term['i']
        j = f_term['j']
        f_val = f_term['val']
        for d_term in non_zero_D: # Iterate M_D times
            k = d_term['k']
            l = d_term['l']
            d_val = d_term['val']

            # Ensure i, j, k, l are valid and correspond to defined x variables
            # This check might be redundant if x is defined for all n,n
            coefficient = f_val * d_val
            if coefficient != 0: # Final check for product being zero
                q_coeffs.append(coefficient)
                q_vars1.append(x[i,k])
                q_vars2.append(x[j,l])

    if q_coeffs: # If there are any quadratic terms
        obj.addTerms(q_coeffs, q_vars1, q_vars2)


    model.setObjective(obj, GRB.MINIMIZE)

    print('\nBeginning Optimization\n')
    model.optimize()
