import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

fileName = "tai256c.dat"

readFile = open(fileName, "r")
data = readFile.read().split()

print(data)

N = int(data[0])

data = data[1:]

D = np.zeros((N,N))
F = np.zeros((N,N))

for r in range(N):
    for c in range(N):
        F[r,c] = int(data[r*N+c])
        D[r,c] = int(data[N*N+r*N+c])
n = N

# PARAMETERS FROM https://optimization-online.org/wp-content/uploads/2023/12/FKKMSR1.pdf
model = gp.Model("QAP")
model.params.SoftMemLimit = 40 # GB  
model.setParam("NodefileStart", 40.0)
model.params.MIPFocus=3
model.params.PreQLinearize=0
model.Params.Threads = 20
model.Params.LogFile = "tai256c_log.txt"


x = model.addVars(n, vtype=GRB.BINARY, name="x")

# Constraint:
model.addConstr(gp.quicksum(x[i] for i in range(n)) == 92, name=f"assign_facility")

# Objective function
quad_obj = gp.QuadExpr()

for i in range(n):
    for j in range(n):
        coeff = D[i,j]
        quad_obj.addTerms(coeff, x[i], x[j]) 

model.setObjective(quad_obj, GRB.MINIMIZE)

print('\nBeginning Optimization\n')
model.optimize()