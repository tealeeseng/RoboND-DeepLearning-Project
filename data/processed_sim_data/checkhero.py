import os
import glob
from scipy import misc
import numpy as np

img_dir = "/home/leeseng/Projects/RoboND/Module6-DeepLearning/RoboND-DeepLearning-Project/data/processed_sim_data/train/masks"
total_files, total_hero = 0, 0
os.chdir(img_dir),
for file in glob.glob("*.png"):
  total_files +=1
  img = misc.imread(file, flatten=False, mode='RGB')
  blue = img[:,:,2]

  if np.sum(blue == 255)> 0 and np.sum(blue <100):
    total_hero += 1
   
percent_hero = 100. * total_hero / total_files
    
print (percent_hero, "percent of files contain the hero")
