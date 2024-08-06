import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import pickle
import os

class WeatherModel:
    def __init__(self, data_path='rag_search/seattle-weather.csv'):
        self.data_path = data_path
        self.model = None
        self.le = None
        self.model_accuracy = None  # To store model accuracy


    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        self.le = LabelEncoder()
        df['weather_label'] = self.le.fit_transform(df['weather'])
        return df
    
    def handle_query(self, query, intent, model_name):
            if intent == "0" and (model_name == "gaussianNB" or model_name == "Gaussian Naive Bayes"):
                # 处理基于天气条件的预测查询
                # predict weather condition with precipitation 10.9, temp_max 10.6, temp_min 2.8, wind 4.5
                pattern = r"predict weather condition with precipitation (\d+(\.\d+)?), temp_max (\d+(\.\d+)?), temp_min (\d+(\.\d+)?), wind (\d+(\.\d+)?)"
                match = re.search(pattern, query)
                if match:
                    precipitation = float(match.group(1))
                    temp_max = float(match.group(3))
                    temp_min = float(match.group(5))
                    wind = float(match.group(7))
                    predicted_weather, accuracy_info = self.predict_weather(precipitation, temp_max, temp_min, wind)
                    return model_name, predicted_weather, accuracy_info
                else:
                    return "Error", "Invalid query format for weather prediction.", "N/A"
            elif intent == "1":
                # 处理基于日期的查询
                date_query = self.extract_date_from_query(query)
                if date_query:
                    temp_info = self.show_temperature_by_date(date_query)
                    return "traditional database query", temp_info, "N/A"
                else:
                    return "Error", "Invalid date format or no data for the given date.", "N/A"

    def train_model_and_save(self):
        df = self.load_data()
        X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
        y = df['weather_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = GaussianNB().fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        self.save_model()

    def save_model(self):
        with open('weather_model.pkl', 'wb') as f:
            pickle.dump((self.model, self.model_accuracy, self.le), f)
        print("Model and accuracy saved successfully.")

    def load_model(self):
        if os.path.exists('weather_model.pkl'):
            with open('weather_model.pkl', 'rb') as f:
                self.model, self.model_accuracy, self.le = pickle.load(f)
            print("Model and accuracy loaded successfully.")
        else:
            print("Model file not found, training new model.")
            self.train_model_and_save()

    def predict_weather(self, precipitation, temp_max, temp_min, wind):
        if not self.model:
            self.load_model()
        weather_label = self.model.predict([[precipitation, temp_max, temp_min, wind]])[0]
        predicted_weather = self.le.inverse_transform([weather_label])[0]
        accuracy_info = f"Model Accuracy: {self.model_accuracy:.2f}"
        return predicted_weather, accuracy_info

    def show_temperature_by_date(self, date_query):
        print(date_query)
        df = self.load_data()
        df.set_index('date', inplace=True)
        if date_query in df.index:
            temp_info = f"Max Temp: {df.loc[date_query]['temp_max']}, Min Temp: {df.loc[date_query]['temp_min']}"
            return temp_info
        else:
            return "No data for the given date or invalid date format."
        
    @staticmethod
    def extract_date_from_query(query):
        # 匹配不同的日期格式：YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD, 或 YYYY.MM.DD
        match = re.search(r'(\d{4})[-/\.]?(\d{2})[-/\.]?(\d{2})', query)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(year=int(year), month=int(month), day=int(day)).date()
            except ValueError:
                return None
        else:
            return None


if __name__ == "__main__":
    weather_model = WeatherModel()
    choice = input("Enter '1' to predict weather based on conditions, '2' to extract and show temperature by date: ")

    if choice == '1':
        precipitation = float(input("Enter precipitation: "))
        temp_max = float(input("Enter maximum temperature: "))
        temp_min = float(input("Enter minimum temperature: "))
        wind = float(input("Enter wind speed: "))
        predicted_weather, accuracy_info = weather_model.predict_weather(precipitation, temp_max, temp_min, wind)
        print("Predicted Weather:", predicted_weather)
        print(accuracy_info)
    elif choice == '2':
        date_query = input("Enter date (YYYY-MM-DD): ")
        date = WeatherModel.extract_date_from_query(date_query)
        if date:
            temp_info = weather_model.show_temperature_by_date(date)
            print(temp_info)
        else:
            print("Invalid date format.")
    else:
        print("Invalid choice.")
