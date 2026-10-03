import os
import subprocess

data_path = "qapdata"
solution_path = "qapsoln"

file_names = []

nonOpt = ['sko42.dat','sko49.dat','sko56.dat','sko64.dat','sko72.dat','sko81.dat','sko90.dat','sko100a.dat','sko100b.dat','sko100c.dat','sko100d.dat','sko100e.dat','sko100f.dat']
nonOpt += ['tai30a.dat','tai35a.dat','tai35b.dat','tai40a.dat','tai40b.dat','tai50a.dat','tai50b.dat','tai60a.dat','tai60b.dat','tai64c.dat','tai80a.dat','tai80b.dat','tai100a.dat','tai100b.dat','tai150b.dat','tai256c.dat']
nonOpt += ['tho40.dat','tho150.dat']
nonOpt += ['wil50.dat','wil100.dat']
nonOpt = ['tai20a.dat','tai25a.dat','tai30a.dat','tai35a.dat','tai40a.dat','tai50a.dat','tai60a.dat','tai80a.dat','tai100a.dat','tai20b.dat','tai25b.dat','tai30b.dat','tai35b.dat','tai40b.dat','tai50b.dat','tai60b.dat','tai80b.dat','tai100b.dat','sko42.dat','sko49.dat','sko56.dat','sko64.dat','sko72.dat','sko81.dat','sko90.dat','sko100a.dat','sko100b.dat','sko100c.dat','sko100d.dat','sko100e.dat','sko100f.dat']


try:
    for item in sorted(os.listdir(data_path)):
        item_path = os.path.join(data_path, item)
        if os.path.isfile(item_path):
            file_names.append(item)
except FileNotFoundError:
    print(f"Error: Folder not found at '{data_path}'")
except Exception as e:
    print(f"An error occurred: {e}")

print(len(file_names))

for trial in range(10):
    ender = ''
    if (trial > 0):
        ender = '_'+str(trial)
    for name in file_names:


        if (os.path.isdir(name+"_0.00_0.40_10r_1000s_w10"+ender) or "tai256c" in name):
            continue
        if(not (name in nonOpt)):
            continue
        bestSolution = 0
        try:
            with open(solution_path+"/"+name[:-3]+"sln", 'r') as file: 
                for line in file:
                    parts = line.strip().split() 
                    bestSolution = int(parts[1])
                    break
        except Exception:
            continue


        al = 0.00
        ah = 0.4
        r = 10
        print(name)
        try:
            result = subprocess.Popen("./qap_gpu"+" ./"+data_path+"/"+name+" --al "+str(al)+" --ah "+str(ah)+" -o " +str(name)+"_"+str("{:.2f}".format(al))+"_"+str("{:.2f}".format(ah))+"_"+str(r)+"r_1000s_w10"+ender+" -s 1000000 --large-energies true --pool-width 10 "+("-g " if name in nonOpt else "-g ")  +str(bestSolution), shell=True,start_new_session=True)
            result.wait(timeout=(1000+60))
        except subprocess.TimeoutExpired:
            print("Process timed out. Cleaning up...")
            # Send a termination signal to the entire process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
