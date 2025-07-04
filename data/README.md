# 📁 `/data` Folder – Required Data Files

This folder does **not include raw data files**, as they may be large, subject to license restrictions, or unsuitable for version control.

## 1. INSEE IRIS Shapefiles (Geometries of IRIS zones)

**Official source**:  
All IRIS shapefiles are available from INSEE via the IGN geoservices portal:  
👉 https://geoservices.ign.fr/irisge#telechargementter2025

### 1.1 France

Download the "France métropolitaine" shapefile for 2025 in GPKG format.  
Then rename the file `iris.gpkg` to:

- `iris_france.gpkg`

### 1.2 Île-de-France Departments

Download the shapefiles for the following departments (75, 77, 78, 91, 92, 93, 94, 95) for the year 2025, each in GPKG format.  
For each, rename the downloaded `iris.gpkg` file to:

- `iris_{department_number}.gpkg`

**Required files**:  
- `iris_75.gpkg`  
- `iris_77.gpkg`  
- `iris_78.gpkg`  
- `iris_91.gpkg`  
- `iris_92.gpkg`  
- `iris_93.gpkg`  
- `iris_94.gpkg`  
- `iris_95.gpkg`  

📁 Place all `.gpkg` files directly in the `/data` folder.

---

## 2. Socioeconomic Data by IRIS (Income, Deciles, etc.)

**Required file**:  
- `BASE_TD_FILO_DEC_IRIS_2020.csv`

**Official source**:  
👉 https://www.insee.fr/fr/statistiques/7233950

📁 Place this `.csv` file directly in the `/data` folder.

---

## 📁 Expected Folder Structure

```
data/
├── iris_75.gpkg
├── iris_77.gpkg
├── iris_78.gpkg
├── iris_91.gpkg
├── iris_92.gpkg
├── iris_93.gpkg
├── iris_94.gpkg
├── iris_95.gpkg
├── base-ic-evol-struct-distrib-revenus-iris-2019.csv
└── README.md
```

---

## 📝 Notes

- These files are not included in the Git repository to avoid versioning large binaries or redistributing official datasets.
- Make sure that the filenames match exactly those expected in the notebooks.
- If you rename or move any file, update the relevant paths in the notebooks or in `utils.py`.

---

