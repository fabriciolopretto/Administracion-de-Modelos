import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from metaflow import FlowSpec, step, S3
import matplotlib.pyplot as plt
from io import BytesIO
import boto3
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
import pickle
import joblib
from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"

# Cargar variables de entorno
load_dotenv()

## Configuración de PostgreSQL
PG_USER = os.getenv('PG_USER', 'metaflow')
PG_PASSWORD = os.getenv('PG_PASSWORD', 'metaflow')
PG_DATABASE = os.getenv('PG_DATABASE', 'metaflow')
PG_PORT = os.getenv('PG_PORT', '5432')
PG_HOST = os.getenv('PG_HOST', 'postgres_sql')  # Cambiado a 'postgres' para usar el nombre del servicio en Docker

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
engine = create_engine(DATABASE_URL)

Session = sessionmaker(bind=engine)

# Definición del modelo
Base = declarative_base()

class Metric(Base):
    __tablename__ = 'metricas'

    id = Column(Integer, primary_key=True)
    model = Column(String)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

# Crear la tabla si no existe
Base.metadata.create_all(engine)

def check_database_connection():
    try:
        with engine.connect() as connection:
            print("Successfully connected to the database.")
    except Exception as e:
        print(f"Error connecting to the database: {str(e)}")
        raise

class ConfusionMatrixFlow(FlowSpec):
    
    @step
    def start(self):
        print("Starting Confusion Matrix Flow")
        self.next(self.load_and_prepare_data)
    
    @step
    def load_and_prepare_data(self):
        data = load_breast_cancer()
        # pylint: disable=no-member
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        # pylint: enable=no-member

        X = df.drop(columns=['target'])
        y = df['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.X_test.to_csv('breast_cancer.csv', index=False)
        self.y_test.to_csv('breast_cancer_y.csv', index=False)
        
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

        self.next(self.train_tree_model, self.train_svc_model, self.train_knn_model, self.train_reglog_model)
    
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

        model_pkl_file = "svc_model.pkl"
        with open(model_pkl_file, 'wb') as file:
            pickle.dump(grid_search_svc, file)
        self.upload_to_s3(model_pkl_file, 'amqtp', model_pkl_file)

        self.next(self.join_models)

    @step
    def train_knn_model(self):
        param_grid_knn = {
            'n_neighbors': [3, 5, 7],  # Número de vecinos
            'weights': ['uniform', 'distance'],  # Peso de los vecinos
            'metric': ['euclidean', 'manhattan']  # Métrica de distancia
        }

        knn = KNeighborsClassifier()
        grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='f1')
        grid_search_knn.fit(self.X_train_scaled, self.y_train)

        self.best_knn_model = grid_search_knn.best_estimator_

        y_pred_knn = self.best_knn_model.predict(self.X_test_scaled)

        self.knn_precision = precision_score(self.y_test, y_pred_knn, pos_label=0)
        self.knn_recall = recall_score(self.y_test, y_pred_knn, pos_label=0)
        self.knn_f1 = f1_score(self.y_test, y_pred_knn, pos_label=0)

        print(f"KNN Model - Precision: {self.knn_precision}, Recall: {self.knn_recall}, F1 Score: {self.knn_f1}")

        # save the iris classification model as a pickle file
        model_pkl_file = "knn_model.pkl"  

        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(grid_search_knn, file)
        self.upload_to_s3(model_pkl_file, 'amqtp', model_pkl_file)

        self.next(self.join_models)

      
    @step
    def train_reglog_model(self):
        param_grid_reglog = {
        'C': [0.1, 1, 10],  # Valores de regularizacion
        'penalty': ['l2'],  # Tipo de regularizacion
        'solver': ['lbfgs']  # Solvers que soportan regularizacion l2
    }

        # Crea el modelo base
        reglog = LogisticRegression(max_iter=10000, random_state=42)

        # Define la busqueda de hiperparametros por grilla con validacion cruzada
        grid_search_reglog = GridSearchCV(reglog, param_grid_reglog, cv=5, scoring='f1')

        # Entrena el modelo con la busqueda de hiperparametros
        grid_search_reglog.fit(self.X_train_scaled, self.y_train)

        # Evalua el mejor modelo en el conjunto de prueba
        self.best_reglog_model = grid_search_reglog.best_estimator_

        # Predice en el conjunto de prueba
        y_pred_reglog = self.best_reglog_model.predict(self.X_test_scaled)

        self.reglog_precision = precision_score(self.y_test, y_pred_reglog, pos_label=0)
        self.reglog_recall = recall_score(self.y_test, y_pred_reglog, pos_label=0)
        self.reglog_f1 = f1_score(self.y_test, y_pred_reglog, pos_label=0)

        print(f"Logistic Regression Model - Precision: {self.reglog_precision}, Recall: {self.reglog_recall}, F1 Score: {self.reglog_f1}")

        # save the reglog model as a pickle file
        model_pkl_file = "reglog_model.pkl"  

        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(grid_search_reglog, file)
        self.upload_to_s3(model_pkl_file, 'amqtp', model_pkl_file)
        self.next(self.join_models)


    @step
    def join_models(self, inputs):
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


         # Merge other necessary attributes
        self.X_test_scaled = inputs[0].X_test_scaled
        self.y_test = inputs[0].y_test

        print("All models joined successfully")
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        self.metrics = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            y_prob = model.predict_proba(self.X_test_scaled)[:, 1]  # Probabilities for the positive class
            
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Maligno (0)', 'Benigno (1)'])
             # pylint: disable=no-member
            fig = disp.plot(cmap=plt.cm.Blues, values_format='d', colorbar=False).figure_
             # pylint: disable=no-member
            plt.title(f'Matriz de Confusión - {model_name.capitalize()}')
            
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            s3 = boto3.client('s3')
            s3.put_object(Bucket='amqtp', Key=f'confusion_matrix_{model_name}.png', Body=buf.getvalue())
            
            plt.close(fig)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc = roc_auc_score(self.y_test, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{model_name.capitalize()} (AUC = {auc:.2f})', color='blue', lw=2)
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title(f'Curva ROC - {model_name.capitalize()}')
            plt.legend(loc="lower right")
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            s3.put_object(Bucket='amqtp', Key=f'roc_curve_{model_name}.png', Body=buf.getvalue())
            
            plt.close()
            
            precision = precision_score(self.y_test, y_pred, pos_label=0)
            recall = recall_score(self.y_test, y_pred, pos_label=0)
            f1 = f1_score(self.y_test, y_pred, pos_label=0)
            
            self.metrics[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        
        print("Metrics:", self.metrics)
        
        self.next(self.store_results)
    
    @step
    def store_results(self):
        print("Storing metrics in PostgreSQL and Redis")
        
        # Print the metrics again to verify they're available in this step
        print("Metrics in store_results:", self.metrics)

        # Almacenar en PostgreSQL usando SQLAlchemy
        session = Session()
        
        for model_name, metrics in self.metrics.items():
            new_metric = Metric(
                model=model_name,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1']
            )
            session.add(new_metric)
        
        session.commit()
        session.close()
        
        self.next(self.end)
    
    @step
    def end(self):
        print("Finished Confusion Matrix Flow")

        # Query to check if data was saved
        session = Session()
        results = session.query(Metric).all()
        count = 0
        if results:
            for result in results:
                print(count, result.model, result.precision, result.recall, result.f1_score)
                count += 1
        else:
            print("No data found")


    def upload_to_s3(self, file_name, bucket, object_name=None):
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(file_name, bucket, object_name or file_name)
            print(f'File uploaded successfully to {bucket}/{object_name}')
        except Exception as e:
            print(f"Error uploading the file: {str(e)}")

if __name__ == '__main__':
    check_database_connection()
    ConfusionMatrixFlow()