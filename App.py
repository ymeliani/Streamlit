import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error

data = pd.read_csv('housing_dataset.csv', sep=',')

#Fonction pour compter les valeurs manquantes dans le dataset
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)
    print("Your selected dataframe has {} columns.".format(df.shape[1]) + '\n' + 
    "There are {} columns that have missing values.".format(mis_val_table_ren_columns.shape[0]))
    return mis_val_table_ren_columns

missing_values_table(data)

#Gestion NaN 
#On remplace les NaN par "NO" car le NaN correspond à l'absence de ces caractéristiques
for column in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType',
              'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType']:
    data[column] = data[column].fillna('NO')
    
#On remplace les NaN par 0
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

#Normalisation LotFrontage
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())

label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col].astype(str))

# Sélectionner les colonnes utiles

#df = data[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd']]

df = data[["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Neighborhood", "OverallQual", "YearBuilt",
          "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "TotRmsAbvGrd", "GarageCars",
          "PoolQC"]]

# Supprimer les valeurs manquantes
df.dropna(inplace=True)

# Diviser les données en ensembles d'entraînement et de test
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
accuracy = r2_score(y_test, y_pred)
print("La précision du modèle (Regression Linéaire) est de :", accuracy)

# Calculer la MAE
mae = np.mean(np.abs(y_pred - y_test))
print("MAE : {:.2f}".format(mae))

# Calculer la MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("MAPE : {:.2f}%".format(mape))

# Créer et entraîner le modèle
regr =  RandomForestRegressor()

# Effectuer la cross-validation avec 5 folds
scores = cross_val_score(model, X_train, y_train, cv=8, scoring='neg_mean_absolute_percentage_error')

# Afficher la MAPE moyenne et l'écart-type
mape_scores = -scores
print("MAPE (Cross Validation): %0.2f (+/- %0.2f)" % (mape_scores.mean(), mape_scores.std() * 2))

regr.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred_rf = regr.predict(X_test)

# Calculer la précision du modèle
accuracy = r2_score(y_test, y_pred_rf)
print("La précision du modèle (Random Forest) est de :", accuracy)

# Calculer la MAE
mae = np.mean(np.abs(y_pred_rf - y_test))
print("MAE : {:.2f}".format(mae))

# Calculer la MAPE
mape = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
print("MAPE : {:.2f}%".format(mape))


# Définition de la fonction pour prédire le prix
def predict_price(inputs):
    inputs = pd.DataFrame([inputs])
    prediction = regr.predict(inputs)
    return prediction[0]


def main():
    st.title("Prédiction de prix de maison")

    # Création des widgets pour l'entrée des données
    mssubclass = st.slider("MSSubClass", min_value=20, max_value=190, step=10)
    mszoning = st.selectbox("MSZoning", ["RL", "RM", "FV", "RH"])
    lotfrontage = st.number_input("LotFrontage", min_value=0, max_value=200)
    lotarea = st.number_input("LotArea", min_value=0, max_value=100000)
    neighborhood = st.selectbox("Neighborhood", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"])
    overallqual = st.slider("OverallQual", min_value=1, max_value=10, step=1)
    yearbuilt = st.number_input("YearBuilt", min_value=1800, max_value=2023)
    yearremodadd = st.number_input("YearRemodAdd", min_value=1800, max_value=2023)
    totalbsmtsf = st.number_input("TotalBsmtSF", min_value=0, max_value=6000)
    firstflrsf = st.number_input("1stFlrSF", min_value=0, max_value=6000)
    secondflrsf = st.number_input("2ndFlrSF", min_value=0, max_value=6000)
    grlivarea = st.number_input("GrLivArea", min_value=0, max_value=8000)
    totrmsabvgrd = st.slider("TotRmsAbvGrd", min_value=1, max_value=15, step=1)
    garagecars = st.slider("GarageCars", min_value=0, max_value=10, step=1)
    poolqc = st.selectbox("PoolQC", ["NA", "Ex", "Fa", "Gd"])

    # Stockage des données en dictionnaire pour prédiction
    inputs = {
        "MSSubClass": mssubclass,
        "MSZoning": mszoning,
        "LotFrontage": lotfrontage,
        "LotArea": lotarea,
        "Neighborhood": neighborhood,
        "OverallQual": overallqual,
        "YearBuilt": yearbuilt
    }