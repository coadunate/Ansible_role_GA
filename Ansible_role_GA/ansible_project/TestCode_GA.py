#!/usr/bin/env python
# Load libraries
import sys
import pandas as pd
import numpy as np
import csv
from pyevolve import GSimpleGA, G1DList, GAllele, Initializators, Mutators, DBAdapters, Selectors, Consts
from sklearn.feature_selection import SelectKBest, f_classif
#from astropy.io import ascii


#load the dataset. The test data have 500 features and 10 samples
tab = pd.read_csv("/path/to/the/data", header=0, index_col=0)
table=np.transpose(tab)
Names = list(table.columns)

# Create features and classes
#features in this case are OTUs. That is,
f = table[:]
#class of each sample
c = [1,1,1,1,1,2,2,2,2,2]

#Let's define fitness function. Create an SelectKBest object to select features with best ANOVA F-Values. Use f_classif to get the ANPVA F-values between label/feature for selection/classification tasks.
#This can be modified according to the research problem and the input data. For example, ANOVA can be replaced by Chi2, mutual information, etc. Also, you can define your own fitness function, in that case the fitness function part of the code has o be rewritten.
fvalue_selector = SelectKBest(f_classif, k=300)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(f, c)

#create the new set of features using the score obtained from fvalue_selector.
X_new = fvalue_selector.transform(f)
print(fvalue_selector.get_support(indices=True))

### Show results
print('Original number of features:', f.shape[1])
print('Reduced number of features:', f_kbest.shape[1])
print(f_kbest[0:11])
print(f_kbest.shape)
print(type(f_kbest))

#convert f_kbest to a dataframe.
df = pd.DataFrame(f_kbest)

#get the scores for the selected features
scores = fvalue_selector.scores_

#Let's get the selected feature names
feature_names = list(f.columns[fvalue_selector.get_support(indices=True)])
features = str(feature_names)

f = open('/path/to/the/fitness/function/result', 'w')
f.write(features)
f.close()

#let's define a function with these selected features to use in the genetic algorithm
#This function takes in consortium (group of OTUs) and needs to return a numerical score for the number of OTUs that are found in feature_names
def eval_func(chromosome):
   return sum([1 for otu in chromosome if otu in feature_names])

# Our genetic algorithm based method is going to consist of a single allele that can be made up of any combination of OTUs.
allele = GAllele.GAlleleList(Names)        #list of all OTUs present in the dataset.
alleles = GAllele.GAlleles([allele], homogeneous=True)
genome = G1DList.G1DList(30)                # set the consortium size, let's use 30 as the size.
genome.setParams(allele=alleles)            # make sure we are choosing from the whole list of OTUs in the dataset.
genome.evaluator.set(eval_func)             # This is where we use the fitness function to compare the OTUs from the data to the one which is selected after applying the fitness function, in this case ANOVA.
genome.mutator.set(Mutators.G1DListMutatorAllele)  # We need to use the mutator function to apply mutation to the consortiums
genome.initializator.set(Initializators.G1DListInitializatorAllele)  # and initializor
GSimpleGA.GSimpleGA(genome, seed=None, interactiveMode=True)   #set the genetic algorithm, use the module GsimpleGA from pyevolve
ga = GSimpleGA.GSimpleGA(genome) #apply genetic algorithm
ga.setPopulationSize(100)                   #set the population size, this is a genetic algorithm parameter that is totally problem dependant.
ga.setGenerations(100)                      #run for 100 generations, this is a genetic algorithm parameter that is totally problem dependant.
ga.setCrossoverRate(0.8)                    #set the crossover rate, this a genetic algorithm parameter that is totally problem dependant.
ga.setMutationRate(0.009)                   #set the mutation rate, this is a genetic algorithm parameter that is totally problem dependant.


#Let's start the evaluation.
print('*'*100)
print("starting evalution")
csv_adapter = DBAdapters.DBFileCSV(identify="run1",filename="/path/to/statistics/output",frequency=1)
ga.setDBAdapter(csv_adapter)
ga.evolve(freq_stats=40)  # start and print stats every 40 generations
population = ga.bestIndividual() # get the selected features.
best = ga.getPopulation()
p =str(population)


#write the output(resluted consortium) to a file named GAoutput.txt
f = open('/path/to/GAoutput.txt', 'w')
f.write(p)
f.close()
