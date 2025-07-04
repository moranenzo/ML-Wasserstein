# 📁 `/data` Folder – Required Data Files

This folder does **not include raw data files**, as they may be large, subject to license restrictions, or unsuitable for version control.

## 1. INSEE IRIS Shapefiles (Geometries of IRIS zones)

**Official source**:
All the IRIS Shapefiles come from the INSEE website, you can find them at the INSE's open portal https://geoservices.ign.fr/irisge#telechargementter2025

### 1.1 France:

Download the "France métropolitaine" from 2025 in GPKG format and rename iris.gpkg into iris_france_gpkg.

**Required file:**
- `iris_france.gpkg`  


### 1.2 Department of Ile-de-France:

Download the IRIS shapefiles for departments 75, 77, 78, 91, 92, 93, 94 and 95 (departments of Ile-de-France) from 2025 in GPKG format and rename the iris.gpkg of each folder into iris_{number of the department}. You should have all those 8 GPKG documents : 

**Required files**:  
- `iris_75.gpkg`  
- `iris_77.gpkg`  
- `iris_78.gpkg`  
- `iris_91.gpkg`  
- `iris_92.gpkg`  
- `iris_93.gpkg`  
- `iris_94.gpkg`  
- `iris_95.gpkg`  

**How to organize**: Place all `.gpkg` files directly in the `/data` directory.

---

### 2. Socioeconomic Data by IRIS (Income, Deciles, etc.)

**Required file**:  
- `BASE_TD_FILO_DEC_IRIS_2020.csv`

**Official source**:  
https://www.insee.fr/fr/statistiques/7233950

**How to organize**: Place the `.csv` file in the `/data` directory.

---

## 📁 Expected Folder Structure

