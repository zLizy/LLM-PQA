import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re
import pickle
import os

class LinearRegressionModel:
    def __init__(self):
        self.mongo_uri = "mongodb+srv://arslan:771944972@cluster0.qv2ymat.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        self.db_name = "sample_analytics"
        self.collection_name = "transactions"
        self.model = None

    def connect_to_mongodb(self):
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]
        return collection

    def fetch_data(self, user_id):
        collection = self.connect_to_mongodb()
        data = []
        for item in collection.find({"account_id": user_id}):
            for transaction in item['transactions']:
                data.append({
                    "account_id": item['account_id'],
                    "transaction_count": item['transaction_count'],
                    "bucket_start_date": item['bucket_start_date'],
                    "bucket_end_date": item['bucket_end_date'],
                    "transaction_date": transaction['date'],
                    "amount": transaction['amount'],
                    "transaction_code": transaction['transaction_code'],
                    "symbol": transaction['symbol'],
                    "price": float(transaction['price']),
                    "total": float(transaction['total'])
                })
        return pd.DataFrame(data)
    
    # 在LinearRegressionModel类中添加

    def handle_query(self, query, intent, model_name):
        user_id = self.extract_user_id_from_query(query)
        if user_id is None:
            return "Error", "Cannot find user ID in the query", "N/A"
        if intent == "0" and model_name == "linear regression":  # 预测下一次交易金额
            df = self.fetch_data(user_id)
            if df.empty:
                return "Error", "No data found for the user.", "N/A"
            if self.model is None:
                self.load_model()  # 确保模型已经加载
            last_transaction_price = df.iloc[-1]['price']
            prediction_result = self.predict(last_transaction_price)
            prediction_accuracy_info = str(self.evaluate_model(df))
            return model_name, str(prediction_result), prediction_accuracy_info
        elif intent == "1":  # 读取用户上一次的交易金额
            last_transaction_amount = self.fetch_last_transaction_amount(user_id)
            return "traditional database query", last_transaction_amount, "N/A"


    def fetch_last_transaction_amount(self, user_id):
        collection = self.connect_to_mongodb()
        result = collection.find_one({"account_id": user_id}, sort=[("transactions.date", -1)])
        if result and "transactions" in result and len(result["transactions"]) > 0:
            last_transaction = result["transactions"][0]
            return last_transaction.get("amount", "Transaction amount not found")
        else:
            return "Transaction amount not found"

    def train_model(self, df):
        X = df[['price']]
        y = df['amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        print("Model trained successfully.")
        self.save_model()

    def predict(self, price):
        if self.model is not None:
            return int(self.model.predict(np.array([[price]]))[0])
        else:
            raise Exception("Model is not trained.")

    def evaluate_model(self, df):
        print("使用了新class")
        X_test = df[['price']]
        y_test = df['amount']
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        accuracy_info = f"R^2 score: {r2:.2f}"

        # return r2_score(y_test, y_pred)
        return accuracy_info

    def save_model(self):
        with open("linear_regression_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        print("Model saved successfully.")

    def load_model(self):
        if os.path.exists("linear_regression_model.pkl"):
            with open("linear_regression_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("Model loaded successfully.")
        else:
            print("Model does not exist. Please train the model first.")

    @staticmethod
    def extract_user_id_from_query(query):
        matches = re.findall(r'\b\d+\b', query)
        if matches:
            return int(matches[-1])
        else:
            return None

if __name__ == "__main__":
    lr_model = LinearRegressionModel()
    lr_model.load_model()
    user_id = input("Please enter a user ID to query: ")  # 改为输入以适应示例
    user_id = LinearRegressionModel.extract_user_id_from_query(user_id) if user_id.isdigit() else None  # 确保用户输入有效
    if user_id:
        df = lr_model.fetch_data(user_id)
        if df.empty:
            print("No data found for the user.")
        else:
            if lr_model.model is None:
                lr_model.train_model(df)
            last_transaction_price = df.iloc[-1]['price']
            prediction = lr_model.predict(last_transaction_price)
            print(f"Predicted next transaction amount: {prediction}")
            last_transaction_amount = lr_model.fetch_last_transaction_amount(user_id)
            print(f"Last transaction amount: {last_transaction_amount}")
    else:
        print("Invalid user ID provided.")
