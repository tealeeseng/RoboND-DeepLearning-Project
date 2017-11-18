import os
import glob
from scipy import misc
import numpy as np

img_dir = "/home/leeseng/Projects/RoboND/Module6-DeepLearning/RoboND-DeepLearning-Project/data/processed_sim_data/train/masks"
total_files, total_hero = 0, 0
os.chdir(img_dir),
for file in glob.glob("*.png"):
  total_files +=1
  names = file.split(".")
  names[0].replace("mask","cam1")
  img = misc.imread(file, flatten=False, mode='RGB')
  blue = img[:,:,2]

  if np.any(blue == 255):
    total_hero += 1
  else:
    names = file.split(".")
    name="../images/"+names[0].replace("mask","cam1")+".jpeg"
    os.remove(name)
    os.remove(file) 
    print(name)
   
percent_hero = 100. * total_hero / total_files
    
print (percent_hero, "percent of files contain the hero")
