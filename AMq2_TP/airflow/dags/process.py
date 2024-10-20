from airflow.decorators import dag, task
import datetime
import pickle
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=60),
    'is_active': True
    
}

@dag(
    dag_id="process_model_cancer_mama",
    description="Modelo de clasificación - Tumores de cáncer de mama.",
    tags=["Modelo", "Cáncer de mama", "Tumores"],
    default_args=default_args,
    catchup=False,
    schedule_interval='0 * * * *',  # Corre cada hora
    start_date=datetime.datetime(2024, 10, 21),
    is_paused_upon_creation=False
)

def process_model_cancer_mama():


    @task.virtualenv(
        task_id="get_data",
        requirements=["scikit-learn==1.2.2", "pandas~=2.0"],
        system_site_packages=True
    )

    def get_data():
        """
        Carga los datos desde de la fuente
        """
        data = load_breast_cancer()

        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        path = "./data.csv"
        df.to_csv(path, index=False)


        return path


	
    @task.virtualenv(
        task_id="split_data",
        requirements=["scikit-learn==1.2.2", "pandas~=2.0"],
        system_site_packages=True,
        multiple_outputs=True
    )

    def split_data(path):
        """
        Divide el data set en entrenamiento y test
        """ 

        if path:

            df = pd.read_csv(path)
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Guardar los conjuntos de entrenamiento y prueba
            X_train_path = "./X_train.csv"
            X_test_path = "./X_test.csv"
            y_train_path = "./y_train.csv"
            y_test_path = "./y_test.csv"

            X_train.to_csv(X_train_path, index=False)
            X_test.to_csv(X_test_path, index=False)
            y_train.to_csv(y_train_path, index=False)
            y_test.to_csv(y_test_path, index=False)
        
            return {"X_train_path": X_train_path, "X_test_path": X_test_path, 
                    "y_train_path": y_train_path, "y_test_path": y_test_path}



    @task.virtualenv(
        task_id="train_model",
        requirements=["scikit-learn==1.2.2", "pandas~=2.0"],
        system_site_packages=True        
    )

    def train_model(df_paths):
        """
        Entrenar modelo
        """ 
        scaler = StandardScaler()
        X_train = pd.read_csv(df_paths['X_train_path'])
        X_test = pd.read_csv(df_paths['X_test_path'])
		
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()

        y_train = pd.read_csv(df_paths['y_train_path'])

        model.fit(X_train_scaled, y_train)
		
        df_paths['path_model'] = './model.pkl'
        with open(df_paths['path_model'], 'wb') as f:
            pickle.dump(model, f)

        df_paths['path_scaler'] = './scaler.pkl'
        with open(df_paths['path_scaler'], 'wb') as f:
            pickle.dump(scaler, f)

        return df_paths
    
    
    
    @task.virtualenv(
        task_id="eval_model",
        requirements=["scikit-learn==1.2.2", "pandas~=2.0"],
        system_site_packages=True  
    )
    
    def eval_model(df_paths):
        """
        Evaluar modelo
        """   	
        with open(df_paths['path_model'], 'rb') as f:
            model = pickle.load(f)
		
        X_test_scaled = pd.read_csv(df_paths['X_test_scaled'])
        y_pred = model.predict(X_test_scaled)
		
        y_test = pd.read_csv(df_paths['y_test_path'])

        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión")
        cm_path = "./confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        df_paths['cm_path'] = cm_path

        return df_paths
	
 
    data_path = get_data()
    df_paths = split_data(data_path)
    df_paths = train_model(df_paths)
    eval_model(df_paths)
    
    
dag = process_model_cancer_mama()
