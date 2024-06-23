import pandas as pd
from dineof import dineof

data = pd.read_csv('data/jester-data-1.csv', header=None, na_values=99)
data = data.drop(columns=0)

data_matrix = data.to_numpy()
result = dineof(data_matrix)

print(result["Xa"])