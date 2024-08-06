import pandas as pd
import numpy as np
import pickle
import os
from openai import OpenAI
import key_param
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import load_data,load_database

class LinearRegressionModel:
    def __init__(self):
        self.filepath = None
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.query = None
        self.dataset_name = None
        self.mse = None
        self.r2 = None
        self.ai_client = OpenAI(api_key=key_param.OPENAI_API_KEY)
        self.encoder = None

    def set_filepath(self, filepath):
        self.filepath = filepath
        
    def load_data(self):
        if self.filepath is None:
            raise ValueError("Filepath is not set.")
        self.data = pd.read_csv(self.filepath, on_bad_lines='skip')
        self.data.columns = self.data.columns.str.strip().str.replace('"', '')
        print("Data loaded successfully.")
        self.encode_categorical_columns()
        self.data.fillna(method='ffill', inplace=True)
        self.data.dropna(inplace=True)
        print(self.data.head())
    
    def handle_query(self, query, model_name, dataset_name, use_new):
        self.query = query
        self.dataset_name = dataset_name
        file_path = f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_files\\{dataset_name}.csv'
        # file_path = f'rag_search\database_files\{dataset_name}.csv'
        self.set_filepath(file_path)

        if use_new:
            self.load_data()
            self.choose_columns()  # Ensures features are set
            self.train_model()
            performance_metrics = {'MSE': self.mse, 'R2': self.r2}
            model_name = self.generate_model_name()
            model_profile = self.generate_model_profile(model_name)
            dataset_profile = self.generate_dataset_profile()
            self.save_model(model_name, model_profile, dataset_profile, performance_metrics)
            
            # load_data.upload_and_move_files()
            # load_database.upload_and_move_files()

        else:
            self.load_data()
            self.load_model(model_name)
            performance_metrics = {'MSE': self.mse, 'R2': self.r2}

        if self.model and self.features:
            print("Please enter the required values for prediction:")
            # result = self.predict()
            result = self.predict_from_query(query)
        else:
            result = "Error: Model not trained or features not set."
            print("Error: Model not trained or features not set.")

        return model_name, result, performance_metrics
    
    def encode_categorical_columns(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        if not categorical_columns.empty:
            self.encoder = OneHotEncoder(sparse_output=False)
            encoded_data = self.encoder.fit_transform(self.data[categorical_columns])
            # 创建编码后的DataFrame 
            encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(categorical_columns))
            # 删除原始分类列并合并编码后的列
            self.data.drop(categorical_columns, axis=1, inplace=True)
            self.data = pd.concat([self.data, encoded_df], axis=1)
        print("Categorical columns encoded if any.")
        print(self.data.head())

    def choose_columns(self):
        print("Available columns in the dataset: ", self.data.columns.tolist())

        suggested_columns, suggested_target = self.suggest_columns_for_model(self.query)
        if suggested_columns and suggested_target:
            print(f"Suggested input features -> {suggested_columns}, output target -> {suggested_target}")
            # Automatically use suggested columns without asking the user
            self.features = suggested_columns
            self.target = suggested_target
            print(f"Features and target set: {self.features}, {self.target}")
        else:
            print("AI's suggestion was invalid or not applicable. Please enter the column names manually.")
            # Manual entry as a fallback
            self.features = input("Please enter the column names for input variables (comma separated): ").split(',')
            self.target = input("Please enter the column name for output variable: ").strip()
            if all(feature in self.data.columns for feature in self.features) and self.target in self.data.columns:
                print(f"Features and target set: {self.features}, {self.target}")
            else:
                print("Error: One or more specified columns are invalid. Please try again.")
                self.features, self.target = None, None  # Reset and continue the loop

    # def choose_columns(self):
    #     print("Available columns in the dataset: ", self.data.columns.tolist())

    #     while True:
    #         suggested_columns, suggested_target = self.suggest_columns_for_model(self.query)
    #         if suggested_columns and suggested_target:
    #             print(f"Suggested input features -> {suggested_columns}, output target -> {suggested_target}")
    #             response = input("Would you like to use these columns? (yes/no): ").lower()
    #             if response == 'yes':
    #                 # 清洗列名以去除任何非数据列的前缀或格式
    #                 # cleaned_features = [col.split(': ')[-1] for col in suggested_columns]  # 仅获取冒号后的部分
    #                 cleaned_features = [col.strip().strip("'").replace("Input variables: ", "").replace("Output variable: ", "") for col in suggested_columns]
    #                 cleaned_target = suggested_target.strip().strip("'").replace("Input variables: ", "").replace("Output variable: ", "")

    #                 # if all(feature in self.data.columns for feature in cleaned_features) and suggested_target in self.data.columns:
    #                 if all(feature in self.data.columns for feature in cleaned_features) and cleaned_target in self.data.columns:
    #                     self.features = cleaned_features
    #                     self.target = cleaned_target
    #                     print(f"Features and target set: {self.features}, {self.target}")
    #                     return
    #                 else:
    #                     print("Error: One or more specified columns are invalid. Please try again.")
    #             else:
    #                 print("Please enter the column names manually.")
    #                 self.features = input("Please enter the column names for input variables (comma separated): ").split(',')
    #                 self.target = input("Please enter the column name for output variable: ").strip()
    #                 if all(feature in self.data.columns for feature in self.features) and self.target in self.data.columns:
    #                     print(f"Features and target set: {self.features}, {self.target}")
    #                     return
    #                 else:
    #                     print("Error: One or more specified columns are invalid. Please try again.")
    #                     self.features, self.target = None, None  # Reset and continue the loop
    #         else:
    #             print("AI's suggestion was invalid or not applicable. Please enter the column names manually.")
    #             self.features = input("Please enter the column names for input variables (comma separated): ").split(',')
    #             self.target = input("Please enter the column name for output variable: ").strip()
    #             if all(feature in self.data.columns for feature in self.features) and self.target in self.data.columns:
    #                 print(f"Features and target set: {self.features}, {self.target}")
    #                 return  # Valid column names set
    #             else:
    #                 print("Error: One or more specified columns are invalid. Please try again.")
    #                 self.features, self.target = None, None  # Reset and continue the loop



    def suggest_columns_for_model(self, query):
        if not query:
            raise ValueError("Query must not be empty for suggesting columns.")
        
        # prompt = (
        #     f"Given the columns {self.data.columns.tolist()} in a dataset and a user's query related to regression analysis, "
        #     f"the user's query is: '{self.query}'. The task involves predicting a numerical outcome based on various input features. "
        #     f"Based on the user's query and the available columns, please suggest the most appropriate column names for:"
        #     f"1. Input variables: columns that will serve as input features for predicting the outcome."
        #     f"2. Output variable: the single column that represents the target outcome to predict."
        #     f"Please provide the column names for input variables as a list and the output variable as a single name, all separated by a comma. "
        #     f"For example, if the task is to predict house prices based on features like age, location, and size, respond with 'age,location,size,house price'."
        #      ['age', 'bmi', 'children', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest', 'charges']
        # )

        prompt = f"""
            Given the columns {self.data.columns.tolist()} in a dataset and a user's query related to regression analysis, 
            the user's query is: '{self.query}'. The task involves predicting a numerical outcome based on various input features. 
            Based on the user's query and the available columns, please suggest the most appropriate column names for: 
            1. Input variables: columns that will serve as input features for predicting the outcome. 
            2. Output variable: the single column that represents the target outcome to predict. 
            Please suggest all relevant columns as a single list in the order they should be used for modeling, with the column representing the target outcome last, all separated by a comma.
            For example, for a task to predict insurance charges based on a person's details, if the dataset columns are ['age', 'bmi', 'children', 'charges', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'],
            you should respond with: 'age', 'bmi', 'children', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest', 'charges'.
            please do not reponse like:" ["Input variables: 'age'", " 'bmi'", " 'children'", " 'sex_female'", " 'smoker_no'", " 'region_northeast'  \nOutput variable: 'charges'"]", only reply with variables names like the example given above.
            """


        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        # Parsing the response to extract suggested columns
        answer = response.choices[0].message.content.strip().split(',')

        print("LLM关于推荐列的回复是：", answer)

        suggested_features = [feat.strip().strip("'").strip() for feat in answer[:-1]]
        suggested_target = answer[-1].strip().strip("'").strip()  # The last one is the target
        print("Suggested input features:", suggested_features, "Suggested output target:", suggested_target)
        
        return suggested_features, suggested_target


    def train_model(self):
        if not self.features or not self.target:
            raise ValueError("Features or target not set.")

        X = self.data[self.features]
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")

        # Model Evaluation
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.mse = mse
        self.r2 = r2
        print("均方误差 (MSE):", mse)
        print("R² 分数:", r2)

    def predict(self):
        if self.model is None:
            raise Exception("Model not trained.")
        if not self.features:
            raise Exception("No features to predict with. Ensure features are set properly.")

        # User input for prediction
        try:
            print('特征打印出来是：', self.features)
            user_input = np.array([[float(input(f"Enter value for {feature}: ")) for feature in self.features]])

            print('用户输入的内容是：', user_input)
            predicted_value = self.model.predict(user_input)
            print(f"Predicted value: {predicted_value[0]:.2f}")
            return predicted_value
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

    def predict_from_query(self, query):
        # Assume that features are defined for the model
        extracted_features = self.extract_features_with_llm(query, self.features)

        if extracted_features is None:
            print("Feature extraction failed.")
            return None

        input_features = np.array([extracted_features])

        print(f"Input features (array shape {input_features.shape}): {input_features}")

        try:
            predicted_value = self.model.predict(input_features)
            print(f"Predicted value: {predicted_value}")

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

        return predicted_value
    

    def save_model(self, model_name, model_profile, dataset_profile, performance_metrics):
        model_filename = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}.pkl"
        model_info_filename = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}_info.pkl"
        model_metrics_filename = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}_metrics.pkl"

        model_profile_filename = f"{model_name}_profile.txt"
        dataset_profile_filename = f"{self.dataset_name}.txt"

        # Save the model
        with open(model_filename, "wb") as f:
            pickle.dump(self.model, f)

        # Save the features and target
        with open(model_info_filename, "wb") as f:
            pickle.dump({'features': self.features, 'target': self.target}, f)
        with open(model_metrics_filename, "wb") as f:
            pickle.dump(performance_metrics, f)

        with open(os.path.join('D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_profile_files', model_profile_filename), 'w') as file:
            file.write(model_profile)

        with open(os.path.join('D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_profile_files', dataset_profile_filename), 'w') as file:
            file.write(dataset_profile)

        print(f"Model and its info saved as {model_filename} and {model_info_filename}")

    def load_model(self, model_name):
        # model_filename = f"rag_search/model_files/{model_name}.pkl"
        model_filename = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}.pkl"
        model_info_filename = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}_info.pkl"
        model_metrics_filename = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}_metrics.pkl"

        print(f"Trying to load model from: {model_filename}")

        if os.path.exists(model_filename):
            with open(model_filename, "rb") as f:
                self.model = pickle.load(f)
            
            with open(model_info_filename, "rb") as f:
                info = pickle.load(f)
                self.features = info['features']
                self.target = info['target']
            with open(model_metrics_filename, "rb") as f:
                performance_metrics = pickle.load(f)
                self.mse = performance_metrics.get('MSE', None)
                self.r2 = performance_metrics.get('R2', None)
                print("Model loaded successfully. Performance Metrics:")
                print(performance_metrics)
            
            print("Model and its info loaded successfully.")
        else:
            print("Model file or info file does not exist.")
            raise FileNotFoundError("Model file or info file does not exist.")



    def generate_model_name(self):
        # Prompt for generating a model name
        prompt = f"""
        Given the dataset with columns {self.data.columns.tolist()} and using input features {self.features} with target {self.target}, this model will perform linear regression analysis based on the user's query: '{self.query}'.
        Please generate a concise and descriptive model name that includes the term 'linear_regression' and clearly reflects its purpose. The model name should:
        1. Not exceed 30 characters in length.
        2. Include only the model name, without spaces or special characters, and be suitable for filenames.
        For example, a suitable model name for a task predicting house prices based on location and size could be 'house_price_linear_regression'. 
        A suitable model name for a task predicting student's performance index could be 'performance_linear_regression'. 
        Please only reply with the model name.
        """

        # Sending the prompt to the LLM
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": self.query}
            ]
        )
        model_name = response.choices[0].message.content.strip().replace(' ', '_').replace(',', '').lower()
        if 'regression' not in model_name:
            model_name += '_regression'
        print(f"Suggested model name: {model_name}")
        return model_name
    
    def generate_model_profile(self, model_name):
        # Setting up the prompt for the LLM to generate the model profile
        prompt = f"""
        Create a detailed model profile for '{model_name}' based on the following specifications:

        - Model Name: {model_name}

        - Dataset Name: {self.dataset_name}

        - Model Overview: A linear regression model designed to predict outcomes based on numerical inputs. This model, utilizes input features {self.features} to predict the target {self.target} as influenced by the user's query: '{self.query}'.

        -Intended Use: This model is intended for use in sectors like real estate, finance, or any field where predicting continuous outcomes is valuable. It helps in making informed decisions by providing estimates based on historical data inputs.

        - Technical Details:
            Algorithm Type: Linear Regression
            Input Features: {self.features}
            Output: Predicted value of {self.target}

        - Model Performance:
            Mean Squared Error (MSE): {self.mse},
            R² Score: {self.r2}

        - Limitations:
            Linear regression assumes a linear relationship between input variables and the target. It may perform poorly if this assumption is violated or if the data contains high multicollinearity or outliers.
        """

        # Sending the prompt to the LLM
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": self.query}
            ]
        )
        model_profile = response.choices[0].message.content.strip()
        print("Generated Model Profile:")
        print(model_profile)
        return model_profile
    
    def generate_dataset_profile(self):
        # Example attributes to include in the dataset profile
        dataset_name = self.dataset_name  # Extract just the file name
        column_names = self.data.columns.tolist()
        sample_data = self.data.head().to_string(index=False)  # Convert the first few rows into a string format without index

        # Setting up the prompt for the LLM to generate the dataset profile
        prompt = f"""
        Create a detailed dataset profile for '{self.dataset_name}' based on the following specifications:

        - Dataset Name: {self.dataset_name}

        - Overview: This dataset contains data structured in several columns: {column_names}. Below is a sample of the data to provide insight into the typical content and structure of the dataset:

            {sample_data}

        - Usage: This dataset is primarily used for building predictive models in the {dataset_name.split('_')[0]} domain. 
        """

        # Sending the prompt to the LLM
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        dataset_profile = response.choices[0].message.content.strip()
        print("Generated Dataset Profile:")
        print(dataset_profile)
        return dataset_profile
    
    def extract_features_with_llm(self, query, features):
        prompt = (
            f"Given the user query: '{query}', extract the following features: {', '.join(features)}. Provide the extracted values in the order they should be used for modeling. "
            f"For example, for a query 'predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children', and features are ['sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest', 'age', 'bmi', 'children'], you should reply with '1, 0, 0, 1, 1, 0, 0, 0, 19, 27.9, 0'."
        )
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        extracted_features = response.choices[0].message.content.strip().replace("'", "")

        print(f"Raw LLM output for features: {extracted_features}")

        try:
            feature_values = [float(value.strip()) for value in extracted_features.split(',')]
            print(f"Parsed feature values: {feature_values}")
        except ValueError as e:
            print(f"Error parsing features: {e}")
            return None  

        return feature_values

    



if __name__ == "__main__":
    regression_model = LinearRegressionModel()
    regression_model.query = 'I want to make a real estate price prediction'
    regression_model.set_filepath('rag_search\database_files\Real_estate.csv')
    regression_model.load_data()
    regression_model.choose_columns()
    regression_model.train_model()
    regression_model.predict()

    model_name = regression_model.generate_model_name()
    model_profile = regression_model.generate_model_profile(model_name)
    regression_model.save_model(model_name,model_profile,dataset_profile=None)




    


# predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children

# predict real estate price with transaction date 2012.917,  house age 32, distance to the nearest MRT station 84.87882, number of convenience stores 10, latitude 24.98298, longitude 121.54024

# predict a student's Performance index with 7 hours studied, previous scores 99, extracurricular activities yes, sleep hours 9 and 1 sample question papers practiced




# predict a student's Performance index with 3 hours studied, previous scores 60, extracurricular activities no, sleep hours 10 and 0 sample question papers practiced

