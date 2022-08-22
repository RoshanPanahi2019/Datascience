# Visualizating the corelation between the project delay features and the project construction complete delay. 

from tarfile import DEFAULT_FORMAT
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from os.path import exists
import numpy as np
path="/media/ms/D/myGithub_Classified/Skanska/Data_Source/Out/Merge_Tbl_1_3_Freq_Reduced_Label_OneHotEncoded.csv"
df=pd.read_csv(path)
row,column=df.shape

# legend={}
# for clm in range(column):
#     new="F"+str(clm)
#     entry={new:[df.columns[clm]]}
#     legend.update(entry)
#     df = df.rename(columns={df.columns[clm]: new})

# legend = pd.DataFrame.from_dict(legend).T # transpose to look just like the sheet above

# legend_path="./legend.xlsx"
# if not exists(legend_path):legend.to_excel('legend.xlsx') # store legend for crelation matrix.
corrMatrix = df.corr()
   
heat_map_path="./corelation_matrix_Freq_Label.png"
if not exists(heat_map_path): 
    sn.set(rc = {'figure.figsize':(12,12)})
    mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
    cmap = sn.diverging_palette(230, 20, as_cmap=True)
    heat_map=sn.heatmap(corrMatrix , mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Project Delay Root-Cause Corelation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    heat_map = heat_map.get_figure() 
    heat_map.savefig(heat_map_path, dpi=400)
print(df.sum())
df.sum().plot()
plt.show()

# Ditribution of the target values
#max=df["F27"].max()
#min=df["F27"].min()
#df["F27"].plot.hist(bins=100, alpha=0.5)
#plt.show() 