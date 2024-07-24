
# Cargar archivo
import pandas as pd
descriptores = "df_con_descriptores_tratado.csv"
df = pd.read_csv(descriptores)

# Calcular descriptores
from rdkit import Chem
from rdkit.Chem import Descriptors

# Calcular objeto mol
def calcular_mol(Smiles):
    molecula = Chem.MolFromSmiles(Smiles)
    return(molecula)

# Agregar columna "mol"
df.loc[:, "mol"] = df["SMILES"].apply(calcular_mol)

# Calcular descriptores
def calcular_descriptores(mol):
    
    mol = Descriptors.CalcMolDescriptors(mol)
    return pd.Series(mol)

df_descriptores = df['mol'].apply(calcular_descriptores)

# Guardar el total de descriptores calculados
df_descriptores.to_csv("descriptores calculados.csv")

# Filtro de baja varianza
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
df_filtrado = selector.fit_transform(df_descriptores)
caracteristicas_seleccionadas = df_descriptores.columns[selector.get_support()]
des_final = df_descriptores[caracteristicas_seleccionadas]

# Matriz de correlación previo al filtro
correlation_matrix = des_final.corr().abs()
threshold = 0.8

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(heatmap)
plt.title('Matriz de Correlación')
plt.xlabel('Características')
plt.ylabel('Características')
plt.show()

# Umbral de correlación
threshold = 0.8  # Por ejemplo, considerar correlaciones superiores a 0.8 como altas

high_correlation_pairs = []

# Iterar sobre la matriz de correlación
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

for feature1, feature2, corr_value in high_correlation_pairs:
    
    # Eliminar la característica feature2
    if feature2 in des_final.columns:
        des_final.drop(columns=[feature2], inplace=True)

# Matriz de correlación
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(heatmap)
plt.title('Matriz de Correlación')
plt.xlabel('Características')
plt.ylabel('Características')
plt.show()

df_final = pd.concat((df, des_final), axis=1)

df_final.to_csv("df_con_descriptores_tratado.csv", index=False)