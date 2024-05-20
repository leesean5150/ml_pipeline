import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

load_dotenv()

def parse_env_list(env_var: str) -> list:
    return os.getenv(env_var, "").split(",")

class DataProcessor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
    
    def load_data(self) -> pd.DataFrame:
            df = pd.read_sql_query("SELECT * FROM calls", self.engine)
            return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        non_unique_ids = df['ID'].value_counts()[df['ID'].value_counts() > 1].index
        non_unique_df = df[df['ID'].isin(non_unique_ids)].drop_duplicates(subset='ID', keep='first')
        df = pd.concat([df[~df['ID'].isin(non_unique_ids)], non_unique_df])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['Call Type'] = df['Call Type'].replace('Whats App', 'WhatsApp')
        return df

class ModelPipeline:
    def __init__(self, categorical_features: list, numeric_features: list, algorithm: str):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.algorithm = self._select_algorithm(algorithm)
        self.pipeline = self._create_pipeline()
    
    def _select_algorithm(self, algorithm_name: str):
        if algorithm_name == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=int(os.getenv("GRADIENT_NUMBER_OF_DECISIONS_TREES")),
                learning_rate=float(os.getenv("GRADIENT_LEARNING_RATE"))
            )
        elif algorithm_name == "RandomForest":
            return RandomForestClassifier(n_estimators=int(os.getenv("NUMBER_OF_TREES")))
        else:
            return DecisionTreeClassifier()
    
    def _create_pipeline(self) -> Pipeline:
        categorical_processor = Pipeline(
            steps=[
                ("imputation_error", SimpleImputer(strategy=os.getenv("CATEGORICAL_IMPUTER_STRATEGY", "most_frequent"))),
                ("onehot", OneHotEncoder(handle_unknown=os.getenv("ONEHOT_HANDLE_UNKOWN", "ignore")))
            ]
        )
        numeric_processor = Pipeline(
            steps=[
                ("imputation_constant", SimpleImputer(
                    missing_values=np.nan,
                    strategy="constant" if os.getenv("NUMERIC_IMPUTER_STRATEGY") == "0" else os.getenv("NUMERIC_IMPUTER_STRATEGY"),
                    fill_value=0
                )),
                ("scaler", StandardScaler())
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("categorical", categorical_processor, self.categorical_features),
                ("numeric", numeric_processor, self.numeric_features)
            ]
        )
        if (os.getenv("ADD_SMOTE") == "True"):
            return ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', self.algorithm)
                ]
        )
        return Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', self.algorithm)
                ] 
        )

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, n_runs: int):
        accuracy_total = 0
        precision_total = 0
        conf_matrix_total = np.array([[0, 0], [0, 0]])
        
        for _ in range(n_runs):
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_test)
            accuracy_total += accuracy_score(y_test, y_pred)
            precision_total += precision_score(y_test, y_pred, pos_label="Scam")
            conf_matrix_total += confusion_matrix(y_test, y_pred, labels=["Scam", "Not Scam"])
        
        accuracy_avg = accuracy_total / n_runs
        precision_avg = precision_total / n_runs
        conf_matrix_avg = conf_matrix_total // n_runs
        
        return accuracy_avg, precision_avg, conf_matrix_avg

def main():
    os.environ['PYTHON_ENV'] = 'development'
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '../data/calls.db')
    
    processor = DataProcessor(db_path)
    df = processor.load_data()
    df = processor.preprocess_data(df)
    
    X = df.drop(columns=["Scam Call"])
    y = df["Scam Call"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(os.getenv("TEST_SIZE")), random_state=42)
    
    categorical_features = parse_env_list("CATEGORICAL_FEATURES")
    numeric_features = parse_env_list("NUMERIC_FEATURES")
    
    model_pipeline = ModelPipeline(categorical_features, numeric_features, os.getenv("ALGORITHM"))
    accuracy_avg, precision_avg, conf_matrix_avg = model_pipeline.train_and_evaluate(X_train, X_test, y_train, y_test, int(os.getenv("NUMBER_OF_RUNS")))
    
    print("Average accuracy of the model with the provided configurations is: {:.2f}%".format(accuracy_avg * 100))
    print("Average precision of the model with the provided configurations is: {:.2f}%".format(precision_avg * 100))
    print("Confusion Matrix of the model with the provided configurations is as follows:")
    print("                    Actual Scam   Actual Not Scam")
    print("Predicted Scam          {}            {}".format(conf_matrix_avg[0][0], conf_matrix_avg[0][1]))
    print("Predicted Not Scam      {}            {}".format(conf_matrix_avg[1][0], conf_matrix_avg[1][1]))

if __name__ == "__main__":
    main()