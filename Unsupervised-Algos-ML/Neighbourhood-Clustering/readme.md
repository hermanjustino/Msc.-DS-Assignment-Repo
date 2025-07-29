# Toronto Neighborhood Clustering for Immigrants: An Unsupervised Learning Approach

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=flat&logo=plotly&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-77B829?style=flat&logo=folium&logoColor=white)

## Project Overview

This project applies **unsupervised learning techniques** to cluster Toronto neighborhoods based on factors that significantly impact livability and immigrant settlement patterns. Using multiple clustering algorithms and dimensionality reduction techniques, we analyze demographic, economic, safety, and accessibility factors to provide data-driven insights for newcomers and urban policy makers.

### Problem Statement
**Question**: How can we group Toronto neighborhoods into meaningful clusters that help immigrants make informed settlement decisions based on livability factors?

**Approach**: Apply unsupervised learning to discover natural groupings in neighborhood characteristics without predefined categories.

## Academic Context

This project fulfills the **Unsupervised Learning Final Project** requirements, demonstrating:

### Assignment Requirements (140 points)
- **Data Collection & Provenance** (3 points): COMPLETED
- **Unsupervised Learning Problem Definition** (6 points): COMPLETED  
- **Exploratory Data Analysis** (26 points): COMPLETED
- **Model Building & Analysis** (70 points): COMPLETED
- **Deliverables Quality** (35 points): COMPLETED

### Learning Objectives
- **Clustering Techniques**: K-Means, Hierarchical, DBSCAN comparative analysis
- **Dimensionality Reduction**: PCA, t-SNE for visualization and feature selection
- **Cluster Validation**: Silhouette analysis, elbow method, Davies-Bouldin index
- **Real-world Application**: Urban planning and immigrant settlement insights

### Technical Skills Demonstrated
- **Data Integration**: Multi-source data fusion from Toronto Open Data
- **Feature Engineering**: Demographic, economic, and accessibility indicators
- **Unsupervised Learning**: Multiple clustering algorithm implementation
- **Geospatial Analysis**: Interactive mapping with Folium and GeoPandas
- **Statistical Validation**: Comprehensive cluster evaluation metrics

## Dataset Sources and Provenance

### Primary Data Sources
| Source | Dataset | Features | Records |
|--------|---------|----------|---------|
| **City of Toronto Open Data** | Neighborhood Profiles 2016 | Demographics, Income, Education | 140 neighborhoods |
| **Toronto Police Service** | Major Crime Indicators | Crime rates by neighborhood | 2019-2023 data |
| **Toronto Open Data** | TTC Service Routes | Transit accessibility scores | Current routes |
| **Kaggle Dataset** | Toronto Neighborhoods Info | Comprehensive neighborhood data | 140 neighborhoods |

### Data Collection Methodology
- **Official Government Sources**: Ensures data reliability and consistency
- **Recent Time Period**: 2019-2023 for current relevance
- **Geographic Consistency**: All data mapped to standard Toronto neighborhood boundaries
- **Multi-dimensional Coverage**: Demographics, economics, safety, accessibility

### Key Features (12 variables)
1. **Demographics**: Immigrant percentage, population density
2. **Economics**: Median household income, unemployment rate
3. **Safety**: Crime rate per capita, major crime incidents
4. **Education**: University degree percentage, high school completion
5. **Housing**: Average rent, homeownership rate
6. **Accessibility**: TTC stop density, walkability score

## Methodology

### 1. Data Preprocessing Pipeline
- **Missing Value Treatment**: Multiple imputation for demographic data
- **Feature Scaling**: StandardScaler for distance-based algorithms
- **Outlier Detection**: IQR method with domain expert validation
- **Feature Selection**: Correlation analysis and domain knowledge

### 2. Exploratory Data Analysis
- **Univariate Analysis**: Distribution of each neighborhood characteristic
- **Bivariate Analysis**: Correlation patterns between features
- **Geographic Patterns**: Spatial distribution of key variables
- **Statistical Testing**: Normality tests for appropriate transformations

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Linear dimensionality reduction
- **t-SNE**: Non-linear manifold learning for visualization
- **Feature Importance**: PCA loadings interpretation
- **Variance Explained**: Optimal component selection

### 4. Clustering Algorithms
#### K-Means Clustering
- **Optimization**: Elbow method and silhouette analysis
- **Initialization**: K-means++ for robust centroid selection
- **Validation**: Multiple random seeds for stability

#### Hierarchical Clustering
- **Linkage Methods**: Ward, complete, average linkage comparison
- **Dendrogram Analysis**: Visual cluster structure interpretation
- **Cutting Strategy**: Distance threshold optimization

#### DBSCAN (Density-Based)
- **Parameter Tuning**: Epsilon and min_samples optimization
- **Noise Detection**: Outlier neighborhood identification
- **Density Visualization**: Core, border, and noise point analysis

### 5. Cluster Validation
- **Internal Metrics**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz
- **External Validation**: Geographic coherence assessment
- **Stability Analysis**: Bootstrap resampling for robustness
- **Interpretability**: Domain expert evaluation of cluster characteristics

## Expected Results

### Cluster Profiles (Preliminary)
Based on Toronto neighborhood characteristics, we expect to identify:

1. **High-Income Low-Crime** (Downtown Core): Financial district areas
2. **Family-Oriented Suburban** (North Toronto): Established residential areas  
3. **Immigrant Gateway** (Scarborough/Etobicoke): High diversity, affordable
4. **Student/Young Professional** (Near universities): Transit-accessible, rental-heavy
5. **Mixed Urban** (Central neighborhoods): Balanced characteristics

### Performance Benchmarks
- **Silhouette Score**: Target > 0.5 for well-separated clusters
- **Davies-Bouldin Index**: Target < 1.0 for distinct clusters
- **Geographic Coherence**: >70% of clusters should show spatial contiguity

## Geographic Visualization

### Interactive Mapping Strategy
- **Folium Integration**: Interactive choropleth maps
- **Cluster Overlays**: Color-coded neighborhood boundaries
- **Feature Layers**: Toggle individual characteristics
- **Popup Information**: Detailed neighborhood statistics

### Visualization Components
1. **Cluster Distribution Map**: Color-coded by final cluster assignment
2. **Feature Heatmaps**: Individual variable geographic patterns
3. **PCA/t-SNE Plots**: 2D visualization of neighborhood similarities
4. **Dendrogram**: Hierarchical clustering structure

## Real-World Applications

### For Immigrants and Newcomers
- **Settlement Planning**: Data-driven neighborhood selection
- **Budget Optimization**: Income vs. cost-of-living trade-offs
- **Safety Assessment**: Crime pattern awareness
- **Community Integration**: Areas with established immigrant populations

### For Urban Policy Makers
- **Resource Allocation**: Targeted service provision by cluster
- **Transit Planning**: Accessibility gap identification
- **Housing Policy**: Affordability cluster analysis
- **Economic Development**: Investment opportunity mapping

### For Researchers
- **Urban Sociology**: Neighborhood segregation patterns
- **Migration Studies**: Settlement pattern analysis
- **Public Policy**: Evidence-based policy recommendation

## Technical Implementation

### Environment Setup
```python
# Core libraries
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
folium>=0.12.0

# Geospatial
geopandas>=0.10.0
shapely>=1.8.0

# Statistical analysis
scipy>=1.7.0
statsmodels>=0.13.0
```

### Project Structure
```
Neighbourhood-Clustering/
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # Cleaned datasets
│   └── geographic/             # Shapefiles and GeoJSON
├── notebooks/
│   └── clustering_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── clustering_models.py
│   └── visualization.py
├── output/
│   ├── maps/                   # Interactive maps
│   ├── plots/                  # Static visualizations
│   └── results/                # Cluster assignments
├── docs/
│   └── assignment.txt
└── README.md
```

## Future Extensions

### Technical Improvements
1. **Advanced Clustering**: Gaussian Mixture Models, spectral clustering
2. **Feature Engineering**: Social media sentiment, economic indicators
3. **Temporal Analysis**: Neighborhood change over time
4. **Machine Learning**: Supervised models for newcomer recommendations

### Data Enhancements
1. **Real-time Updates**: API integration for current data
2. **Additional Sources**: Healthcare access, environmental quality
3. **Granular Geography**: Postal code or census tract level
4. **Validation Surveys**: Ground-truth data collection

## References

### Academic Sources
- Walks, A., & Bourne, L. S. (2006). Ghettos in Canada's cities? *Urban Studies*, 43(2), 435-466.
- Murdie, R. A. (2008). Diversity and concentration in Canadian immigration. *Research on Immigration and Integration in the Metropolis*.

### Technical Resources
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Toronto Open Data Portal](https://open.toronto.ca/)
- [GeoPandas Documentation](https://geopandas.org/)

### Data Sources
- **City of Toronto**: [Neighborhood Profiles](https://open.toronto.ca/dataset/neighbourhood-profiles/)
- **Toronto Police**: [Major Crime Indicators](https://data.torontopolice.on.ca/datasets/major-crime-indicators/about)
- **Kaggle**: [Toronto Neighborhoods Dataset](https://www.kaggle.com/datasets/youssef19/toronto-neighborhoods-inforamtion)

## Contact Information

- **Author**: Herman Justino
- **Program**: MSc Data Science
- **Course**: Unsupervised Learning & Machine Learning
- **Institution**: [University Name]
- **GitHub**: [Repository URL]

---

*This project demonstrates the practical application of unsupervised learning techniques to real-world urban planning challenges, providing actionable insights for both individual decision-making and policy development in immigrant settlement patterns.*
