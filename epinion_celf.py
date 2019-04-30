from random import uniform, seed
import numpy as np
import time
import time 
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.IndependentCascadesModel as ids
import matplotlib
import random
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
import ndlib.models.epidemics.ThresholdModel as th
from celf import celf
import itertools
from operator import add
import pandas as pd

n=8297

gg=[]


with open("/home/rragarwal//nu/nd/epinions_ps.txt", 'r') as f:
	next(f)
	for line in f:
		k=line.split()
		gg.append([k[0], k[1]])


g = nx.DiGraph()

g.add_edges_from(gg)
celf_output =[]
#for p in [0.0054,0.01,0.05, 0.1, 0.5, 1]:
for p in [0.0072, 0.03]:
	for s in [1,5, 10 ,20, 50, 100]:
		res=celf(g,s,p,1000)
		celf_output.append([p, s,  len(res[0]), sum(res[1]), " ".join(res[0]), sum(res[2])])
		print(celf_output)

df=pd.DataFrame(celf_output)

df.to_csv("/home/rragarwal/gtmodels/epinion_clef_1000.csv")
