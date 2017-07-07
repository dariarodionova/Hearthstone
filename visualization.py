import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

initialData = np.array(pd.read_csv("data.csv").values)

cleanedData = np.delete(initialData, [0,1,3,4,5,6,7,8,9,10], 1)
cleanedData = cleanedData[cleanedData[:,0] != "Unknown",:]

clData = pd.DataFrame(cleanedData)
archetype_counts = clData.groupby(0).size()

sorted = archetype_counts.sort_values() # the most underrepresented archetypes at the top
print(sorted.head(20))

uniqueArchetypes = np.unique(np.array(clData[0]))
print(uniqueArchetypes, "\nShape:", uniqueArchetypes.shape)

allAppearingCards = np.sort(np.unique(cleanedData[:, 1:].flatten()))
print(allAppearingCards)


#-----creating classes
item = np.zeros(uniqueArchetypes.shape)
classes = np.identity(uniqueArchetypes.shape[0])
namedClasses = np.column_stack((uniqueArchetypes, classes))
namedClassesDf = pd.DataFrame(namedClasses)
namedClassesDf.to_csv("NamedClasses.csv")
print(namedClasses)

#-----final DataSet
print(cleanedData)
convertedClasses = np.zeros((cleanedData.shape[0], uniqueArchetypes.shape[0]))
'''for i in range(cleanedData.shape[0]):
    "print(np.argwhere(namedClasses[:,0]==cleanedData[i,0]))
'''