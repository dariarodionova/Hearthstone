import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

initialData = np.array(pd.read_csv("data.csv").values)

cleanedData = np.delete(initialData, [0,1,3,4,5,6,7,8,9,10], 1)
cleanedData = cleanedData[cleanedData[:,0] != "Unknown",:]

cleanedDataDF = pd.DataFrame(cleanedData)
#cleanedDataDF.to_csv("CleanedData.csv")
archetype_counts = cleanedDataDF.groupby(0).size()

sorted = archetype_counts.sort_values() # the most underrepresented archetypes at the top
print(sorted.head(20))

uniqueArchetypes = np.unique(np.array(cleanedDataDF[0]))
print(uniqueArchetypes, "\nShape:", uniqueArchetypes.shape)

allAppearingCards = np.sort(np.unique(cleanedData[:, 1:].flatten()))
print(allAppearingCards)

#-----creating classes
namedClasses = uniqueArchetypes.copy()
namedClassesDf = pd.DataFrame(namedClasses)
namedClassesDf.to_csv("NamedClasses.csv")
print("\nNamed classes :\n", namedClasses)

#-----applying classes
m = cleanedData.shape[0]
numCls = uniqueArchetypes.shape[0]

convertedClasses = np.zeros((m))
for i in range(0,m):
   convertedClasses[i] = np.argwhere(namedClasses[:]==cleanedData[i,0])

convertedClassesDf = pd.DataFrame(convertedClasses)
convertedClassesDf.to_csv("ConvertedClasses.csv")

finalData = np.column_stack((cleanedData[:,1:], convertedClasses))
pd.DataFrame(finalData).to_csv("FinalData.csv", index=False, index_label=False)
print(finalData.shape)