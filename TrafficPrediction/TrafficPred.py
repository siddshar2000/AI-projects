import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("usercode/datafile.csv")
print(f"\nData sample \n {df.head(5)}")

# Remove duplicate
print(f"\nNum of duplicate Values: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"\nData sample after duplicate removal \n {df.head()}")

# Show histogram for traffic volume
fig = plt.figure()

axes = seaborn.histplot(df, x='traffic_volume', kde=True)
fig.add_axes(axes)
plt.title("Traffic volume histogram")

fig.show()

# Transforming date & time so that can injected to ML
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
df['day'] = df['date_time'].dt.day_name()
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['hour'] = df['date_time'].dt.hour
df.drop('date_time', axis=1, inplace=True)
print(f"\nDataset sample after date & time extraction: {df.head()}")

# Convert categorical columns into numerical
df = pd.get_dummies(df, columns=list(set(df.columns) - set(df._get_numeric_data().columns)))

# Create input output
y = df["traffic_volume"]
x = df.drop(columns=["traffic_volume"])
print(f"\nPreprocessed intput sample: {x.head()}")
print(f"\nPreprocessed output sample: {y.head()}")

# Split the training & test data
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.85,shuffle = True,random_state=42)

# Build 3 models
models = {
    'Linear Regression':LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest' : RandomForestRegressor()
}

# Train models
for model_name, model in models.items():
    model.fit(x_train, y_train)

# Eval models
model_metrics = {}
for model_name, model in models.items():
    y_pred = model.predict(x_test)
    r2score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    model_metrics[model_name] = [r2score, mse]

print(f"\nmodel_metrics: {model_metrics}")

df_metrics = pd.DataFrame(model_metrics)
df_metrics = df_metrics.transpose()
df_metrics.columns = ["r2_score", "MSE"]
print(df_metrics)

# Show eval metric
fig2 = plt.figure()
axes2 = seaborn.barplot(y='r2_score', x=df_metrics.index, data=df_metrics)
fig2.add_axes(axes2)
fig2.show()

# Save the models
for model_name, model in models.items():
    joblib.dump(model, "usercode/"+ model_name + ".pk1")

# Load and use the models
for model_name in models.keys():
    file_name = "usercode/"+ model_name + ".pk1"
    loaded_model = joblib.load(file_name)
    y2 = loaded_model.predict(x_test)
    mse = mean_squared_error(y_test, y2)
    print(f"\nModel \033[1m{model_name}\033[0m MSE: {mse:.2f}")

print("Press any key to exit")
x = input()