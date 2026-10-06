\# PDAC Radiomics 365-Day Risk Prediction



\## Project Overview



This project investigates 365-day risk classification and survival analysis in pancreatic ductal adenocarcinoma (PDAC) using clinical data and CT-derived radiomic features.



\## Research Question



Can CT-derived radiomic features provide additional predictive information beyond clinical features for 365-day PDAC risk classification?



\## Dataset



\- Clinical cohort: 138 patients

\- 365-day classification cohort: 128 patients

\- CT-eligible radiomics cohort: 46 patients

\- Initial radiomic features: 107

\- Class 0: survival >= 365 days

\- Class 1: death before 365 days



Patients censored before 365 days were excluded from the binary classification cohort.



\## Project Workflow



Clinical Data

\-> Preprocessing

\-> 365-Day Classification

\-> TCIA Clinical-Imaging Matching

\-> CT + RTSTRUCT

\-> Tumor Segmentation

\-> Radiomic Feature Extraction

\-> Radiomics Quality Control

\-> Clinical / Radiomics / Combined ML

\-> Kaplan-Meier Analysis

\-> Cox Proportional Hazards Analysis



\## Team Contributions



\### Person A - Clinical Classification



\- Clinical data preprocessing

\- Exploratory data analysis

\- 365-day classification dataset creation

\- Naive Bayes classification

\- SMOTENC class balancing

\- Logistic Regression

\- Cross-validation

\- Error analysis

\- Feature importance

\- ROC analysis



\### Person B - Imaging and Survival Analysis



\- TCIA clinical-imaging linkage

\- CT series identification

\- RTSTRUCT inspection

\- Tumor segmentation mask generation

\- CT/segmentation quality control

\- PyRadiomics feature extraction

\- Radiomic feature quality control

\- Clinical vs radiomic vs combined ML experiments

\- Kaplan-Meier survival analysis

\- Clinical Cox proportional hazards analysis

\- Radiomic Cox analysis

\- Combined clinical-radiomic survival modeling



\## Main Classification Result



The Person A Naive Bayes model with SMOTENC achieved:



\- Accuracy: 73.08%

\- Class-1 Recall: 56%

\- F1-score: 0.59

\- ROC-AUC: 0.595



For the CT-eligible cohort, radiomics did not demonstrate clear additional predictive value beyond the available clinical variables under the evaluated ML pipeline.



\## Survival Analysis



The clinical survival dataset contained 138 patients, including 110 observed events and 28 censored observations.



The predefined 365-day groups showed strong Kaplan-Meier separation.



Radiomic Cox analysis selected three radiomic features after correlation filtering and univariate Cox screening.



The combined clinical-radiomic Cox model had the highest apparent C-index, but this result is exploratory and was not independently validated.



\## Important Limitations



\- Small CT/radiomics cohort of 46 patients

\- Small number of positive cases

\- High-dimensional radiomic feature space

\- No external validation

\- Exploratory analysis

\- Results are not clinically validated



\## Repository Structure



```text

ml\_mini\_project/

|

├── data/

|   ├── raw/

|   ├── processed/

|   └── radiomics/

|       ├── metadata/

|       └── features/

|

├── src/

|   ├── personA/

|   └── personB/

|

├── results/

|   ├── personA/

|   └── personB/

|

├── personA\_final\_methodology\_notes.txt

├── personB\_final\_methodology\_notes.txt

├── summary\_personA.txt

├── summary\_personB.txt

└── README.md

