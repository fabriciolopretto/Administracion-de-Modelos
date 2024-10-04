import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from metaflow import FlowSpec, step, S3
import hashlib
import redis
import pickle
import joblib
import boto3

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"

class CombinedThreeModelsTrainingAndBatchProcessing(FlowSpec):

    @step
    def start(self):
        print("Starting Combined Three Models Training and Batch Processing")
        self.next(self.load_and_prepare_data)

    @step
    def load_and_prepare_data(self):
        data = load_breast_cancer()  # pylint: disable=no-member
        df = pd.DataFrame(data.data, columns=data.feature_names) # pylint: disable=no-member
        df['target'] = data.target # pylint: disable=no-member

        X = df.drop(columns=['target'])
        y = df['target']


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.X_test.to_csv('breast_cancer.csv', index=False)
        self.y_test.to_csv('breast_cancer_y.csv',  index=False)
        
        
        self.upload_to_s3('breast_cancer.csv', 'amqtp', os.path.join('data', 'breast_cancer.csv'))
        self.upload_to_s3('breast_cancer_y.csv', 'amqtp', os.path.join('data', 'breast_cancer_y.csv'))

        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        self.scaler = scaler

        scaler_file = "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        self.upload_to_s3(scaler_file, 'amqtp', scaler_file)

        self.next(self.train_tree_model, self.train_svc_model)

    @step
    def train_tree_model(self):
        param_grid_tree = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }

        tree_clf = DecisionTreeClassifier(random_state=42)
        grid_search_tree = GridSearchCV(tree_clf, param_grid_tree, cv=5, scoring='f1')
        grid_search_tree.fit(self.X_train_scaled, self.y_train)

        self.best_tree_model = grid_search_tree.best_estimator_

        y_pred_tree = self.best_tree_model.predict(self.X_test_scaled)

        self.tree_precision = precision_score(self.y_test, y_pred_tree, pos_label=0)
        self.tree_recall = recall_score(self.y_test, y_pred_tree, pos_label=0)
        self.tree_f1 = f1_score(self.y_test, y_pred_tree, pos_label=0)

        print(f"Tree Model - Precision: {self.tree_precision}, Recall: {self.tree_recall}, F1 Score: {self.tree_f1}")

        # save the iris classification model as a pickle file
        model_pkl_file = "tree_model.pkl"  

        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(grid_search_tree, file)
            self.upload_to_s3(model_pkl_file, 'amqtp', model_pkl_file)

        self.next(self.join_models)

    @step
    def train_svc_model(self):
        param_grid_svc = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        svc = SVC(probability=True, random_state=42)
        grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='f1')
        grid_search_svc.fit(self.X_train_scaled, self.y_train)

        self.best_svc_model = grid_search_svc.best_estimator_

        y_pred_svc = self.best_svc_model.predict(self.X_test_scaled)

        self.svc_precision = precision_score(self.y_test, y_pred_svc, pos_label=0)
        self.svc_recall = recall_score(self.y_test, y_pred_svc, pos_label=0)
        self.svc_f1 = f1_score(self.y_test, y_pred_svc, pos_label=0)

        print(f"SVC Model - Precision: {self.svc_precision}, Recall: {self.svc_recall}, F1 Score: {self.svc_f1}")

        # save the iris classification model as a pickle file
        model_pkl_file = "svc_model.pkl"  

        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(grid_search_svc, file)
            self.upload_to_s3(model_pkl_file, 'amqtp', model_pkl_file)

        self.next(self.join_models)


    def upload_to_s3(self, file_name, bucket, object_name=None):
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(file_name, bucket, object_name or file_name)
            print(f'Model uploaded successfully to {bucket}/{object_name}')
        except Exception as e:
            print(f"Error uploading the model: {str(e)}")


    @step
    def join_models(self, inputs):
        # This step joins the parallel model training steps
        self.models = {}
        for input in inputs:
            if hasattr(input, 'best_tree_model'):
                self.models['tree'] = input.best_tree_model
            elif hasattr(input, 'best_svc_model'):
                self.models['svc'] = input.best_svc_model
            elif hasattr(input, 'best_knn_model'):
                self.models['knn'] = input.best_knn_model
            elif hasattr(input, 'best_reglog_model'):
                self.models['reglog'] = input.best_reglog_model

        print("All models joined successfully")
        self.next(self.load_models_from_s3)


    @step
    def load_models_from_s3(self):
        s3 = S3(s3root="s3://amqtp/")
        
        self.loaded_models = {}
        for model_name in ["tree", "svc"]:
            try:
                # Get the S3 object and use the path attribute to open the file
                model_obj = s3.get(f"{model_name}_model.pkl")
                with open(model_obj.path, 'rb') as f:
                    self.loaded_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model successfully from S3.")
            except IndexError as e:
                print(f"IndexError while loading {model_name}: {e}")
                raise
            except Exception as e:
                print(f"General error while loading {model_name}: {e}")
                raise

        print("Models loaded from S3")
        self.next(self.end)


    @step
    def end(self):
        print("Finished Combined Three Models Training and Batch Processing")

if __name__ == "__main__":
    CombinedThreeModelsTrainingAndBatchProcessing()
