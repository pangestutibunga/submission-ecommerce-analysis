# E-Commerce Analysis Dashboard

Dashboard for the **E-Commerce Public Dataset**
built using Streamlit. Analysis period: Januari – Desember 2018.

---

## Dashboard Feature

| Tab | Content |
|---|---|
| **Question 1: Revenue & Review** | Top categories with the highest revenue & average review score below 4 |
| **Question 2: Churn Risk** | Proportion & dsitribution of customers customers at churn risk |
| **RFM Analysis** | Customers segmentation into 8 groups based on Recency, Frequency, and Monetary |

---

## Setup Environemnt

1. Extract the downloaded zip file.
2. Navigate to the folder.

```bash
cd dashboard
```
3. Create a virtual environment
```bash
python -m venv venv
```
4. Activate the virtual environment
```bash
venv\Scripts\activate
```
5. Install the required dependencies
```bash
pip install -r requirements.txt
```
## Run Streamlit App
```bash
streamlit run dashboard.py
```
The Dashboard will automatically open in the browser.