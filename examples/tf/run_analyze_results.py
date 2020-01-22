import os
import numpy as np

path = "/Users/tianqi/desktop/rl_code/garage/examples/tf/data/local/experiment/"

target_columns = ["AverageReturn"]

for target_column in ["AverageReturn"]:
  with open(path+'result_ddpg_pendulm_{}.csv'.format(target_column), 'w', encoding = 'utf-8') as result_writer:
    R = np.zeros((495, 8))
    for i in range(1, 9):
      with open(path+'ddpg_pendulm_0{}/ddpg_pendulm_0{}.csv'.format(i,i), 'r', encoding = 'utf-8') as result_reader:
          for line_no, line in enumerate(result_reader):
            segs = line.rstrip('\n').split(',') 
            l = len(segs)
            if line_no == 0:
              for j, seg in enumerate(segs):
                print(seg)
                if seg == target_column:
                  col = j
                  print(seg, col)
                  break
              continue
            R[line_no-1][i-1]=segs[col]
    print(type(R))
    R_mean = np.mean(R, axis=1)
    R_max = np.max(R, axis=1)
    R_min = np.min(R, axis=1)
    for i in range(495):
      res= ",".join([str(r) for r in R[i]]) + "," + str(R_mean[i]) + "," + str(R_max[i]) + "," + str(R_min[i]) + "\n"
      result_writer.write(res)
    metrics = list(np.mean(R, axis=0)) + list([np.mean(R_mean), np.mean(R_max), np.mean(R_min)])
    res= ",".join([str(r) for r in metrics])
    result_writer.write(res)


for target_column in ["AverageReturn"]:
  with open(path+'jole_ddpg_pendulm_v4_{}.csv'.format(target_column), 'w', encoding = 'utf-8') as result_writer:
    R = np.zeros((495, 5))
    for i in range(1, 6):
      with open(path+'jole_ddpg_pendulm_v4_0{}/jole_ddpg_pendulm_v4_0{}.csv'.format(i,i), 'r', encoding = 'utf-8') as result_reader:
          for line_no, line in enumerate(result_reader):
            segs = line.rstrip('\n').split(',') 
            l = len(segs)
            if line_no == 0:
              for j, seg in enumerate(segs):
                if target_column in seg:
                  col = j
                  print(seg, col)
                  break
              continue
            R[line_no-1][i-1]=segs[col]
    R_mean = np.mean(R, axis=1)
    R_max = np.max(R, axis=1)
    R_min = np.min(R, axis=1)
    
    for i in range(495):
      res= ",".join([str(r) for r in R[i]]) + "," + str(R_mean[i]) + "," + str(R_max[i]) + "," + str(R_min[i]) + "\n"
      result_writer.write(res)
    metrics = list(np.mean(R, axis=0)) + list([np.mean(R_mean), np.mean(R_max), np.mean(R_min)])
    res= ",".join([str(r) for r in metrics])
    result_writer.write(res)
