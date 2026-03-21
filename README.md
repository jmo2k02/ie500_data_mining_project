# Data Nuggets - ie500_data_mining_project

Our repo for the Data Mining Project

## Table of Contents

- [Official Lecture Link](https://www.uni-mannheim.de/dws/teaching/course-details/courses-for-master-candidates/ie-500-data-mining/)
- [Course Overview](#course-overview)
- [Project Requirements](#project-requirements)
- [Our Dataset: Twitch Gamers](#our-dataset-twitch-gamers)
- [Proposed Approach](#proposed-approach)
- [Timeline & Deliverables](#timeline--deliverables)

---

## Course Overview

### Key Lecture Topics

**Data Mining Definition:** A non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data.

**Core Data Mining Tasks:**
1. **Classification** (Predictive) - Assign records to predefined classes
   - Methods: K-Nearest-Neighbors, Decision Trees, Naïve Bayes, SVM, Neural Networks
2. **Regression** (Predictive) - Predict continuous values
3. **Cluster Analysis** (Descriptive) - Find groups of similar data points
4. **Association Analysis** (Descriptive) - Discover frequent itemsets and rules

**The Data Mining Process:**
1. **Selection** - Choose relevant data sources
2. **Preprocessing** - Handle missing values, outliers, duplicates; normalize data
3. **Transformation** - Feature engineering, dimensionality reduction, discretization
4. **Data Mining** - Apply algorithms with proper hyperparameter optimization
5. **Interpretation/Evaluation** - Analyze results, iterate to improve model

**Important Notes:**
- Data preparation takes 70-80% of project time
- Always compare results to baseline (e.g., majority class for classification)
- Use proper evaluation: train/test split or cross-validation
- Try multiple methods and hyperparameter settings
- Perform error analysis to understand model behavior

---

## Project Requirements

### Deliverables

**Team:** 6 students per team

**1. Project Outline** (Due: Sunday, April 12th, 23:59)
- Maximum 4 pages including title page (PDF, using DWS thesis template)
- Must answer:
  1. What is the problem you are solving?
  2. What data will you use? Where will you get it?
  3. How will you solve the problem? (preprocessing steps, algorithms)
  4. How will you measure success? (evaluation method)
  5. What do you expect your results to look like?

**2. Final Project Report** (Due: Sunday, May 17th, 23:59)
- Maximum 12 pages including title/toc and references (max 10 pages content)
- Must include:
  - Application area and goals (0.5 pages)
  - Profile of dataset - structure and size (minimum 1 page)
  - Preprocessing steps
  - Data Mining approaches tried with different parameter settings
  - Evaluation setup and results (minimum 2 pages)
  - Error analysis and comparison to state-of-the-art
  - Declaration of AI tools used

**3. Presentation** (Upload by: Thursday, May 21st, 23:59)
- 10 minutes presentation + 5 minutes discussion
- During exercise slot

**4. Coaching Sessions**
- Every team must attend at least one coaching session (Thursdays)

### Grading
- 75% Written Exam
- 20% Project Report
- 5% Project Presentation

---

## Our Dataset: Twitch Gamers

**Source:** [SNAP Stanford - Twitch Gamers](https://snap.stanford.edu/data/twitch_gamers.html)

**Dataset Size:**
- **Nodes:** 168,114 Twitch users
- **Edges:** 6,797,557 mutual follower relationships
- **Features per node:** 9 attributes

**Node Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| views | Numeric | Total channel view count |
| mature | Binary | Explicit content flag (0/1) |
| life_time | Numeric | Days since account creation |
| created_at | Date | Account creation date |
| updated_at | Date | Last profile update |
| numeric_id | Numeric | Unique node identifier |
| dead_account | Binary | Account status (0=active, 1=inactive) |
| language | Categorical | Broadcaster language (20+ categories) |
| affiliate | Binary | Twitch affiliate status (0/1) |

**Network Properties:**
- Large-scale social network graph
- Mutual follower relationships indicate stronger connections than one-way follows
- Rich node features enable both supervised and unsupervised learning

---

## Proposed Approach

### Problem Statement

We will tackle **multiple data mining tasks** to comprehensively analyze the Twitch gaming community:

1. **Binary Classification:** Predict affiliate status (affiliate vs non-affiliate)
2. **Multi-class Classification:** Predict broadcaster language
3. **Regression:** Predict view count based on network features and account characteristics
4. **Cluster Analysis:** Identify communities of streamers with similar profiles

### Why This Dataset?

- **Ground truth available** for supervised learning (affiliate status, language, etc.)
- **Large scale** - demonstrates ability to work with big data
- **Diverse task types** - covers classification, regression, and clustering
- **Real-world relevance** - insights could inform content strategy for streamers
- **Rich feature space** - combines network features (graph structure) with node attributes

### Data Exploration & Preprocessing

**1. Initial Data Profiling**
- Distribution of classes (affiliate status, languages, mature content)
- Missing value analysis
- Outlier detection (views, life_time)
- Correlation analysis between features

**2. Feature Engineering**
- **Network features:**
  - Node degree (number of connections)
  - Clustering coefficient (network density around node)
  - PageRank score (influence metric)
  - Core number (k-core decomposition)
- **Temporal features:**
  - Account age in days
  - Days since last update
  - Activity rate (views per day alive)
- **Derived features:**
  - Log-transformed views (handle skewness)
  - Binary indicator for high-activity accounts
  - Language family groupings (e.g., European vs Asian languages)

**3. Preprocessing Steps**
- Handle missing values (imputation or removal)
- Normalize numerical features (StandardScaler)
- Encode categorical variables (One-Hot or Label Encoding)
- Address class imbalance if present (SMOTE, class weights)
- Train/validation/test split (60/20/20 or 10-fold cross-validation)

### Proposed Methods

**Task 1: Binary Classification (Affiliate Status)**

| Method | Rationale | Hyperparameters to Tune |
|--------|-----------|-------------------------|
| K-Nearest Neighbors | Baseline, works well with network proximity | k (1-20), distance metric |
| Decision Tree | Interpretable rules, handles mixed data types | max_depth, min_samples_split |
| Random Forest | Ensemble method, robust to overfitting | n_estimators, max_depth, max_features |
| Gradient Boosting (XGBoost) | Often best performance on tabular data | learning_rate, n_estimators, max_depth |
| Logistic Regression | Linear baseline, fast | C (regularization strength) |

**Evaluation Metrics:**
- Accuracy (if balanced)
- Precision, Recall, F1-score (if imbalanced)
- ROC-AUC curve
- Confusion matrix analysis

**Task 2: Multi-class Classification (Language)**
- Same methods as Task 1, adapted for multi-class
- Focus on top 5-10 languages if computational constraints arise
- Evaluation: Accuracy, macro/micro F1-score, confusion matrix

**Task 3: Regression (View Count Prediction)**
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- Evaluation: RMSE, MAE, R² score

**Task 4: Cluster Analysis**
- K-Means (with elbow method for optimal k)
- DBSCAN (identify core/outlier streamers)
- Hierarchical clustering
- Evaluation: Silhouette score, inspect cluster characteristics

### Baseline Comparisons

- **Classification:** Majority class predictor
- **Regression:** Mean/median view count predictor
- **Clustering:** Random cluster assignment

### Iterative Improvement Strategy

1. **Iteration 1:** Simple preprocessing + baseline methods
2. **Iteration 2:** Add network features + tune hyperparameters
3. **Iteration 3:** Advanced feature engineering + ensemble methods
4. **Iteration 4:** Error analysis → targeted improvements

---

## Timeline & Deliverables

| Date | Milestone | Tasks |
|------|-----------|-------|
| **March 22, 23:59** | Team Formation | Finalize team on Google Sheet |
| **March 23-Apr 11** | Data Exploration | • Load and profile data<br>• Generate network features<br>• Initial preprocessing<br>• Draft outline |
| **April 12, 23:59** | **Project Outline Due** | Submit 4-page outline via ILIAS |
| **April 15** | Outline Feedback | Review feedback, attend session if required |
| **April 16-May 10** | Implementation | • Implement all four tasks<br>• Hyperparameter tuning<br>• Document experiments<br>• Attend coaching session |
| **May 11-16** | Report Writing | • Write 12-page report<br>• Create visualizations<br>• Error analysis |
| **May 17, 23:59** | **Final Report Due** | Submit report + code via ILIAS |
| **May 18-20** | Presentation Prep | Create 10-min presentation |
| **May 21, 23:59** | **Presentation PDF Due** | Upload slides to ILIAS |
| **May 22** | Presentations | Present to class (10 min + 5 min Q&A) |

---

## Team Organization

**Communication:**
- Weekly sync meetings (decide schedule)
- Shared drive for code and documents (GitHub + Google Drive)
- Track progress in shared spreadsheet

**Suggested Roles:**
- **Data Preparation Lead** (2 people): Preprocessing, feature engineering
- **Modeling Lead** (2 people): Implement and tune classification/regression models
- **Clustering/Analysis Lead** (1 person): Cluster analysis, network visualization
- **Report/Documentation Lead** (1 person): Coordinate writing, create visualizations

**All members:** Participate in coaching sessions, review each other's work, contribute to final report

---

## Resources

**Textbooks:**
- Pang-Ning Tan et al., "Introduction to Data Mining" (2nd Edition)
- Aurélien Géron, "Hands-on Machine Learning with Scikit-Learn" (2nd/3rd Edition)

**Documentation:**
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)

**Dataset:**
- [Paper (arXiv:2101.03091)](https://arxiv.org/abs/2101.03091)
- [GitHub Repository](https://github.com/benedekrozemberczki/datasets)
