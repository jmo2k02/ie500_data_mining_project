# Data Nuggets - ie500_data_mining_project

Our repo for the Data Mining Project.

See the pipeline in the `/Pipeline` folder. There you will find notebooks named **A** through **D**. Each one contains one step of our pipeline. **Y** contains the Evaluation:
1. A -> Loader
2. B -> Feature engineering
3. C -> Training
4. D -> Hyperparameter tuning
5. Y -> Evaluation

## Table of Contents

- [Official Lecture Link](https://www.uni-mannheim.de/dws/teaching/course-details/courses-for-master-candidates/ie-500-data-mining/)
- [Available Services](#available-services)
- [Course Overview](#course-overview)
- [Project Requirements](#project-requirements)
- [Timeline & Deliverables](#timeline--deliverables)

---

## Available Services

| Service | URL | Description | Credentials |
| ------- | --- | ----------- | ----------- |
| MLFlow  | http://116.203.119.229:5000/ | Experiment tracking, model registry, and artifact logging for managing the ML lifecycle. | user = "data_mining", pw = "admin123_datadays" |
| S3-Storage | nbg1.your-objectstorage.com | Object storage backend for datasets, model artifacts, and other persisted files. | ask Justus |
---

## S3 Helper (Upload/Download/Delete)

There is a small helper class at `s3_storage.py` that loads the repo `.env` automatically and wraps common S3 operations.

Example:

```python
from s3_storage import S3Storage

s3 = S3Storage.from_env()

# upload
s3.upload_file(local_path="local.csv", key="datasets/local.csv")

# download
s3.download_file(key="datasets/local.csv", local_path="./downloads/local.csv")

# delete
s3.delete(key="datasets/local.csv")
```

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
