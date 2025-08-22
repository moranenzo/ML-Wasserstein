# 📁 `/data` Folder – Required Data Files

This folder does **not include raw data files**.

## 1. INSEE IRIS Shapefile (Geometries of IRIS zones)

**Required file** :
- `coutours-iris.gpkg`

**Official source**:  
All IRIS shapefiles are available from INSEE via the IGN geoservices portal:  
👉 https://geoservices.ign.fr/irisge#telechargementter2025

Download the "France métropolitaine" shapefile for 2024 in GPKG format.  


---

## 2. Income Data by IRIS (Income, Deciles, etc.)

**Required file**:  
- `BASE_TD_FILO_DEC_IRIS_2020.csv`

**Official source**: https://www.insee.fr/fr/statistiques/7233950

---

## 3. Average Income per age 

**Required file**:  
- `reve-niv-vie-individu-age-med.xlsx`

**Official source**: https://www.insee.fr/fr/statistiques/2416878

---

## 4. Population per Age by IRIS

**Required file**:
- `base-ic-evol-struct-pop-2020.xlsx`

**Official source**: https://www.insee.fr/fr/statistiques/7704076

Place all these files directly in the `/data` folder.

---

## 📁 Expected Folder Structure

```
data/
├── BASE_TD_FILO_DEC_IRIS_2020.csv
├── base-ic-evol-struct-pop-2020.xlsx
├── contours-iris.gpkg
├── reve-niv-vie-individu-age-med.xlsx
└── README.md
```