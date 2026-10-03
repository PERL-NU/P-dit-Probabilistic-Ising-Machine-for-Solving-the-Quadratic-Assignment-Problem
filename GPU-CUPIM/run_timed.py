import os
import subprocess

data_path = "qapdata"
solution_path = "qapsoln"

file_names = []

nonOpt = ['sko42.dat','sko49.dat','sko56.dat','sko64.dat','sko72.dat','sko81.dat','sko90.dat','sko100a.dat','sko100b.dat','sko100c.dat','sko100d.dat','sko100e.dat','sko100f.dat']
nonOpt += ['tai30a.dat','tai35a.dat','tai35b.dat','tai40a.dat','tai40b.dat','tai50a.dat','tai50b.dat','tai60a.dat','tai60b.dat','tai64c.dat','tai80a.dat','tai80b.dat','tai100a.dat','tai100b.dat','tai150b.dat','tai256c.dat']
nonOpt += ['tho40.dat','tho150.dat']
nonOpt += ['wil50.dat','wil100.dat']
nonOpt = ['chr12a.dat','had14.dat','chr15a.dat','esc16a.dat','nug17.dat','chr18a.dat','els19.dat','chr20a.dat','nug21.dat','chr22a.dat','nug24.dat','chr25a.dat','bur26a.dat','nug27.dat','nug28.dat','kra30a.dat','esc32e.dat','tai35a.dat','ste36a.dat','lipa40a.dat','sko42.dat','sko49.dat','lipa50a.dat','sko56.dat','lipa60a.dat','sko64.dat','lipa70a.dat','sko72.dat','lipa80a.dat','sko81.dat','lipa90a.dat','sko100a.dat','esc128.dat','tai150b.dat']


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

for trial in range(1):
    ender = ''
    if (trial > 0):
        ender = '_'+str(trial)
    for name in file_names:

        #if ( "100" in name):
        #    continue
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
            result = subprocess.Popen("./qap_gpu"+" ./"+data_path+"/"+name+" -o " +str(name)+"_"+str("{:.2f}".format(al))+"_"+str("{:.2f}".format(ah))+"_"+str(r)+"r_10000_w10"+ender+" -b 0.0625 -w 10000 --large-energies true --pool-width 10 "+("-g " if name in nonOpt else "-g ")  +str(bestSolution), shell=True,start_new_session=True)
            result.wait(timeout=(1000+60))
        except subprocess.TimeoutExpired:
            print("Process timed out. Cleaning up...")
            # Send a termination signal to the entire process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
