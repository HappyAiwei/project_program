from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from bld.project_paths import project_paths_join as ppj

# Final Prediction:
X = pd.read_csv(ppj("OUT_DATA", "clean_X.csv"))
y = pd.read_csv(ppj("OUT_DATA", "clean_y.csv"))
X_test = pd.read_csv(ppj("OUT_DATA", "clean_X_test.csv"))
test = pd.read_csv(ppj("IN_DATA", "test.csv"))
### Build final estimator.
reg_final = XGBRegressor(max_depth=15, n_jobs=32, n_estimators=100)
reg_final.fit(X, y)

### Adjust column position of test dataset.
a = X_test['Year_2015']
X_test = X_test.drop(['Year_2015'], axis=1)
X_test['Year_2015'] = a

### Run final predict.
final_predictions = reg_final.predict(X_test)

### Final result output!
dic = {'Id':test['Id'].astype(int), 'Sales': np.exp(final_predictions)}
pre_store_sales = pd.DataFrame(dic)

result = pre_store_sales.sort_values(by='Id', ascending=True)
result.to_csv(ppj("OUT_DATA", "prediction_result.csv"), index=False)