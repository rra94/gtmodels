
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.IndependentCascadesModel as ids
import matplotlib
import random
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
import ndlib.models.epidemics.ThresholdModel as th
form celf import celf
import itertools
from operator import add
import pandas as pd

n=8297

gg=[]


with open("/home/rragarwa/Documents/nucleus/nd/wikivote_ps.txt", 'r') as f:
    next(f)
    for line in f:
        k=line.split()
        gg.append([k[0], k[1]])


g = nx.DiGraph()

g.add_edges_from(gg)
celf_output =[]
for p in [0.0072,0.03,0.05, 0.1, 0.5, 1]:
    for s in [1,5, 10,20, 35]:
      celf_output.append([p, s] + celf(g,s, p=,mc=1000)[:2])

df=pd.DataFrame(celf_output)

df.to_csv("/home/rragarwa/gt_models/wikivote_clef_1000.csv")      
