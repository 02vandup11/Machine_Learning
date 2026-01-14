# Customer Segmentation Using Clustering

## Project Description
This project implements **customer segmentation** using **unsupervised machine learning (clustering)** techniques on a real-world marketing campaign dataset. The goal is to identify distinct customer groups based on demographic and purchasing behavior, enabling data-driven marketing strategies and improved customer targeting.

---

## Repository Structure
```
├── Clustering.ipynb          # Jupyter Notebook containing analysis and clustering
├── marketing_campaign.csv   # Dataset used for customer segmentation
└── README.md                # Project documentation
```

---

## Dataset Overview
**File:** `marketing_campaign.csv`

The dataset contains marketing and customer-related data, including:
- Demographic attributes (age, education, income, family size)
- Spending patterns across multiple product categories
- Purchase behavior through various channels
- Marketing campaign responses

This dataset is used to analyze customer behavior and create meaningful customer segments.

---

## Tools & Technologies
- **Python**
- **Jupyter Notebook**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Clustering and preprocessing

---

## Approach & Methodology

### 1. Data Exploration
- Loaded and inspected the dataset
- Analyzed data types and missing values
- Performed basic statistical analysis

### 2. Data Preprocessing
- Handled missing and inconsistent values
- Selected relevant numerical features
- Applied feature scaling for optimal clustering performance

### 3. Clustering
- Implemented **K-Means Clustering**
- Used the **Elbow Method** to determine the optimal number of clusters

### 4. Visualization & Interpretation
- Visualized clusters to understand customer groupings
- Interpreted cluster characteristics based on spending and income patterns

---

## Key Outcomes
- Identified distinct customer segments with similar behavior
- Gained insights into high-value and low-engagement customer groups
- Demonstrated how clustering supports targeted marketing and decision-making

---

## How to Run the Project
1. Clone or download the repository
2. Open `Clustering.ipynb` using Jupyter Notebook
3. Ensure `marketing_campaign.csv` is present in the same directory
4. Run the notebook cells sequentially

---

## Future Improvements
- Experiment with additional clustering algorithms (Hierarchical, DBSCAN)
- Apply dimensionality reduction techniques (PCA)
- Build interactive dashboards using Power BI or Streamlit
- Automate cluster-based recommendations

---

## Author
**Vandana Padhi**  
BSc IT Graduate  
Data Analytics | Machine Learning | AI Enthusiast
