from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
def parse_env_list(env_var):
    return os.getenv(env_var, "").split(",")

engine = create_engine('sqlite:///data/calls.db')
df = pd.read_sql_query("SELECT * FROM calls", engine)

categorical_features = parse_env_list("CATEGORICAL_FEATURES")
numeric_features = parse_env_list("NUMERIC_FEATURES")

non_unique_ids = df['ID'].value_counts()[df['ID'].value_counts() > 1].index
non_unique_df = df[df['ID'].isin(non_unique_ids)]
non_unique_df = non_unique_df.drop_duplicates(subset='ID', keep='first')
df = pd.concat([df[~df['ID'].isin(non_unique_ids)], non_unique_df])

X = df.drop(columns=["Scam Call"])
y = df["Scam Call"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(os.getenv("TEST_SIZE")), random_state=42)

categorical_processor = Pipeline(
    steps=[
        ("imputation_error", SimpleImputer(strategy=os.getenv("CATEGORICAL_IMPUTER_STRATEGY", "most_frequent"))),
        ("onehot", OneHotEncoder(handle_unknown=os.getenv("ONEHOT_HANDLE_UNKOWN", "ignore")))
    ]
)
numeric_processor = Pipeline(
    steps=[
        ("imputation_constant", SimpleImputer(missing_values=np.nan, strategy="constant" if os.getenv("NUMERIC_IMPUTER_STRATEGY") == "0" else os.getenv("NUMERIC_IMPUTER_STRATEGY"), fill_value=0)),
        ("scaler", StandardScaler())
    ]
)
preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_processor, categorical_features),
        ("numeric", numeric_processor, numeric_features)
    ]
)
pipeline_steps = [
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
]

pipeline = Pipeline(pipeline_steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model with the provided configurations is: {:.2f}%".format(accuracy * 100))