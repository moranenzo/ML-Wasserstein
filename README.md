# Wasserstein Clustering of French IRIS

## 📌 Project Overview

This project applies **Wasserstein K-means clustering** to analyze the heterogeneity of French IRIS (statistical units) based on income distributions.
Each IRIS is described by income deciles, which capture the full shape of the distribution instead of reducing it to a single summary statistic such as the mean or median. This distributional perspective enables the identification of richer and more meaningful clusters than those obtained with standard K-means.

We explore two main approaches:

* **Centroid-based Wasserstein K-means (bary\_WKM)**, which builds clusters around Wasserstein barycenters.
* **Distance-based Wasserstein K-means (dist\_WKM)**, which relies on pairwise Wasserstein distances.

Both approaches are inspired by the methodology introduced by Yubo Zhuang et al. (2022).

The project is organized around two main notebooks:

### 📒 `IRIS.ipynb`

This notebook focuses on **income-only distributions**. It applies barycenter-based Wasserstein K-means (bary\_WKM) and compares the outcomes with classical K-means, highlighting the added value of clustering full distributions rather than relying on a single indicator.

### 📒 `IRISxAGE.ipynb`

This notebook extends the analysis to **joint income–age distributions**, providing a more detailed view of population structures. Both bary\_WKM and dist\_WKM are implemented, but the focus is placed on **dist\_WKM**, which yields more stable and interpretable results. The notebook also explores the geographical distribution of clusters, first at the national scale and then with a zoom into Paris.

## 📂 Repository Structure

```
├── data/ 
├── notebooks/
│   ├── IRIS.ipynb
│   └── IRISxAGE.ipynb
├── src/
│   ├── __init__.py             # Marks src as a Python package  
│   ├── utils.py                # Helper functions (normalization, plotting, etc.)  
│   └── clustering_methods.py   # Implementation of clustering methods  
└── README.md 
```

## 📊 Results

* For income-only clustering, bary_WKM revealed negligible differences compared to classical K-means. This is mainly because the median already explains most of the income distribution, while the other deciles only refine the information without creating major differences for clustering.  
* When extending to joint income–age distributions, dist_WKM proved more effective than bary_WKM, which tended to collapse clusters prematurely.  
* Geographical mapping showed limited spatial patterns at the national scale, largely due to the partial coverage of available IRIS data. However, clearer structures emerged when zooming into Paris, where strong differences appeared between clusters obtained at the national level and those derived from a Paris-only clustering.

## 📚 References

* **William, A. (2020). *A Short Introduction to Optimal Transport and Wasserstein Distance*.**  
  This introductory document was the first resource I used to become familiar with Wasserstein distances, which are central to optimal transport. Peyré and Cuturi’s lecture videos also provide excellent introductory material.

* **Zhuang, Y., et al. (2022). *Wasserstein k-means for clustering probability distributions*.**  
  This paper introduces the Wasserstein k-means algorithm and motivates the idea of clustering full distributions instead of reducing them to summary statistics. It provides the conceptual foundation for both the barycenter-based and distance-based approaches explored in this project.

* **Okano, D. & Imaizumi, M. (2024). *Wasserstein k-Centers Clustering for Distributional Data*.**  
  This more recent work extends the field by studying Wasserstein k-centers, offering new perspectives on clustering stability and computational strategies. While not directly implemented in our project, the method proposed by Okano and Imaizumi represents a promising direction for future research.

This project has been supervised by **Vincent Divol (CREST)**.
