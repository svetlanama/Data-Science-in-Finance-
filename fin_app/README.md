

# Instructions to launch
```
pip install -r re
quirements.txt

```
# Instructions to launch
```
cd src

BE: uvicorn  fraud_api:app --reload --port 8502
FE: streamlit run fraud_client.py
```





# App  
![ai-protection.png](images%2Fai-protection.png)

# Feature Importance 
![feature_importance.png](images%2Ffeature_importance.png)

# Prediction results (POSITIVE)
![fraud.png](images%2Ffraud.png)

# Prediction results (NEGATIVE)
![no fraud.png](images%2Fno%20fraud.png)


# SHAP Values Bar Plot

![shap_bar.png](images%2Fshap_bar.png)

# SHAP Values Pie Plot
![shap_pie.png](images%2Fshap_pie.png)


![More Pie Results.png](images%2FMore%20Pie%20Results.png)
# Batch Auto Processing
``` 
source run_stream_scoring.sh

```

# Results
![batch.png](images%2Fbatch.png)