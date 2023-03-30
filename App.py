RUN pip install joblib

import joblib
import streamlit as st
import pandas as pd

# Chargement du modèle
model = joblib.load('nom_du_fichier.joblib')

# Fonction de prédiction
def prediction(model, input_df):
    prediction = model.predict(input_df)
    return prediction

# Interface Streamlit
def main():
    # Titre et description de l'application
    st.title("Prédiction du prix de vente d'une maison")
    st.write("Cet outil permet de prédire le prix de vente d'une maison.")

    # Colonnes du dataset
    col1, col2, col3, col4, col5 = st.beta_columns(5)

    # Colonnes pour les variables explicatives
    with col1:
        ms_subclass = st.number_input("MSSubClass")

    with col2:
        ms_zoning = st.selectbox("MSZoning", options=["RL", "RM", "C (all)", "FV", "RH"])

    with col3:
        lot_frontage = st.number_input("LotFrontage")

    with col4:
        lot_area = st.number_input("LotArea")

    with col5:
        neighborhood = st.selectbox("Neighborhood", options=["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"])

    col6, col7, col8, col9, col10 = st.beta_columns(5)

    with col6:
        overall_qual = st.number_input("OverallQual")

    with col7:
        year_built = st.number_input("YearBuilt")

    with col8:
        year_remod_add = st.number_input("YearRemodAdd")

    with col9:
        total_bsmt_sf = st.number_input("TotalBsmtSF")

    with col10:
        first_flr_sf = st.number_input("1stFlrSF")

    col11, col12, col13, col14, col15 = st.beta_columns(5)

    with col11:
        second_flr_sf = st.number_input("2ndFlrSF")

    with col12:
        gr_liv_area = st.number_input("GrLivArea")

    with col13:
        tot_rms_abv_grd = st.number_input("TotRmsAbvGrd")

    with col14:
        garage_cars = st.number_input("GarageCars")

    with col15:
        pool_qc = st.selectbox("PoolQC", options=["None", "Ex", "Fa", "Gd"])

    # Affichage des valeurs des variables explicatives
    input_dict = {'MSSubClass': ms_subclass, 'MSZoning': ms_zoning, 'LotFrontage': lot_frontage, 
                  'LotArea': lot_area, 'Neighborhood': neighborhood, 'OverallQual': overall_qual, 
                  'YearBuilt': year_built, 'YearRemodAdd': year_remod_add, 'TotalBsmtSF': total_bsmt_sf, 
                  '1stFlrSF': first_flr_sf, '2ndFlrSF': second_flr_sf, 'GrLivArea': gr_liv_area, 
                  'TotRmsAbvGrd': tot_rms_abv_grd, 'GarageCars': garage_cars, 'PoolQC': pool_qc}

    input_df = pd.DataFrame([input_dict])

    # Bouton pour la prédiction
   
