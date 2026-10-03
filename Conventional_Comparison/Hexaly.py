import sys
import os
import subprocess
import hexaly.optimizer
import sys

data_path = "qapdata"
solution_path = "qapsoln"

file_names = []

nonOpt = ['sko42.dat','sko49.dat','sko56.dat','sko64.dat','sko72.dat','sko81.dat','sko90.dat','sko100a.dat','sko100b.dat','sko100c.dat','sko100d.dat','sko100e.dat','sko100f.dat']
nonOpt += ['tai30a.dat','tai35a.dat','tai35b.dat','tai40a.dat','tai40b.dat','tai50a.dat','tai50b.dat','tai60a.dat','tai60b.dat','tai64c.dat','tai80a.dat','tai80b.dat','tai100a.dat','tai100b.dat','tai150b.dat','tai256c.dat']
nonOpt += ['tho40.dat','tho150.dat']
nonOpt += ['wil50.dat','wil100.dat']

def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

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
for name in file_names: #file_names
    if ("tai256c" in name):
        continue

   
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

    if os.path.isfile(name+"_1000.out"):
        continue

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Read instance data
        #
        file_it = iter(read_integers(fileName))

        # Number of points
        n = next(file_it)

        # Distance between locations
        A = [[next(file_it) for j in range(n)] for i in range(n)]
        # Flow between factories
        B = [[next(file_it) for j in range(n)] for i in range(n)]

        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Permutation such that p[i] is the facility on the location i
        p = model.list(n)

        # The list must be complete
        model.constraint(model.eq(model.count(p), n))

        # Create B as an array to be accessed by an at operator
        array_B = model.array(B)

        # Minimize the sum of product distance*flow
        obj = model.sum(A[i][j] * model.at(array_B, p[i], p[j])
                        for j in range(n) for i in range(n))
        model.minimize(obj)

        model.close()
        optimizer.param.set_objective_threshold(0,bestSolution)
        optimizer.param.time_limit = 1000
        optimizer.param.set_nb_threads(20)

        optimizer.solve()

        #
        # Write the solution in a file with the following format:
        #  - n objValue
        #  - permutation p
        #
        with open(name+str("_1000.out"), 'w') as outfile:
            outfile.write("%d %d\n" % (n, obj.value))
            for i in range(n):
                outfile.write("%d " % p.value[i])
            outfile.write("\n")

