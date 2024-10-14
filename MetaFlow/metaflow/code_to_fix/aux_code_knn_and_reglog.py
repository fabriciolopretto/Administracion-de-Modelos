  
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

        print(f"Random Forest Model - Precision: {self.reglog_precision}, Recall: {self.reglog_recall}, F1 Score: {self.reglog_f1}")

        # save the iris classification model as a pickle file
        model_pkl_file = "reglog_model.pkl"  

        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(grid_search_reglog, file)
            self.upload_to_s3(model_pkl_file, 'amqtp', model_pkl_file)
        self.next(self.join_models)