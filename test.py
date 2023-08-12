import os

root_dir = 'D:/Research/Counterfactual/Scripts/'

inpaint_dir = os.path.join(root_dir, 'inpaint')
bg_dir = os.path.join(root_dir, 'bg')

for dirpath, dirnames, filenames in os.walk(bg_dir):
    print(dirpath)
    # structure1 = os.path.join(inpaint_dir, dirpath[len(bg_dir) +1:])
    # if not os.path.isdir(structure1):
    #     os.mkdir(structure1)