import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Recupera datos
df = pd.read_csv("countries.csv")

# Especifica los datos y sus etiquetas
X = df[["Life Expectancy", "GDP Per Capita", "CO2 Emissions Per Capita"]]
y = df["Class"]

# Para visualizar gráfica de dispersión
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df.loc[df["Class"] == "Low", "Life Expectancy"],
    df.loc[df["Class"] == "Low", "GDP Per Capita"],
    df.loc[df["Class"] == "Low", "CO2 Emissions Per Capita"],
    c="purple",
    label="Low"
)

ax.scatter(
    df.loc[df["Class"] == "Lower middle", "Life Expectancy"],
    df.loc[df["Class"] == "Lower middle", "GDP Per Capita"],
    df.loc[df["Class"] == "Lower middle", "CO2 Emissions Per Capita"],
    c="pink",
    label="Lower middle"
)

ax.scatter(
    df.loc[df["Class"] == "Upper middle", "Life Expectancy"],
    df.loc[df["Class"] == "Upper middle", "GDP Per Capita"],
    df.loc[df["Class"] == "Upper middle", "CO2 Emissions Per Capita"],
    c="lightgreen",
    label="Upper middle"
)

ax.scatter(
    df.loc[df["Class"] == "High", "Life Expectancy"],
    df.loc[df["Class"] == "High", "GDP Per Capita"],
    df.loc[df["Class"] == "High", "CO2 Emissions Per Capita"],
    c="darkgreen",
    label="High"
)

# Datos para la predicción
Life_Expectancy = float(input("Indique la esperanza de vida al nacer: "))
GDP_Per_Capita = float(input("Indique el GDP($US): "))
co2 = float(input("Indique las emisiones de CO2 per cápita: "))

dfp = pd.DataFrame({"Life Expectancy": [Life_Expectancy],
                    "GDP Per Capita": [GDP_Per_Capita],
                    "CO2 Emissions Per Capita": [co2]})

ax.scatter(dfp["Life Expectancy"], dfp["GDP Per Capita"], dfp["CO2 Emissions Per Capita"], c="black", marker="x", s=100, label="Data")

# Configuración del gráfico 3D
ax.set_xlabel('Life Expectancy')
ax.set_ylabel('GDP Per Capita')
ax.set_zlabel('CO2 Emissions Per Capita')

ax.legend()

# Calcular k
k = int(input("Dame el valor de K(número impar): "))


knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Predecir y mostrar resultado
prediccion = knn.predict(dfp)
print("\nCon los datos:")
print(dfp)
print("La categoría predicha es:")
print(prediccion)

plt.show()
