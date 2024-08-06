import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model, save_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Concatenate
from keras.models import Model
import tensorflow as tf

import json

from openai import OpenAI
import key_param
import load_data
import os
import re

class Recommender:
    def __init__(self):
        self.filepath = None
        self.data = None
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_col = None
        self.item_col = None
        self.query = None
        self.number_of_users = None
        self.number_of_items = None
        self.train_data = None  
        self.val_data = None 
        self.dataset_name = None
        self.history = None
        self.performance_metrics = None
        self.ai_client = OpenAI(api_key=key_param.OPENAI_API_KEY)

    def set_filepath(self, filepath):
        self.filepath = filepath
        
    def load_data(self):
        if self.filepath is None:
            raise ValueError("Filepath is not set.")
        self.data = pd.read_csv(self.filepath, on_bad_lines='skip')
        self.data.columns = self.data.columns.str.strip().str.replace('"', '')
        print("Data loaded successfully.")
        print(self.data.head())

    # def handle_query(self, query, intent, model_name):
    #     self.query = query
        
    #     # 询问是否使用新的数据路径
    #     use_new_path = input("Using default spotify_dataset.csv. Do you need to provide a new file path? (yes/no): ").lower() == 'yes'
    #     if use_new_path:
    #         new_path = input("Please enter the new file path within the 'rag_search/' directory: ")
    #         self.set_filepath('rag_search/' + new_path)
    #     else:
    #         self.set_filepath('rag_search/spotify_dataset.csv')

    #     self.load_data()
    #     self.prepare_data()

    #     if use_new_path:
    #         # Train a new model since the data path has changed
    #         generated_model_name = self.generate_model_name(query)
    #         model_profile = self.generate_model_profile(query, generated_model_name)
    #         self.build_model()
    #         self.train_model(self.train_data, self.val_data)
    #         self.save_model(generated_model_name, model_profile)
    #         load_data.upload_and_move_files()
    #         self.model = load_model('rag_search/model_files/' + generated_model_name + '.h5')
    #     else:
    #         # Load existing model
    #         self.model = load_model('rag_search/model_files/' + model_name + '.h5')

    #     user_id = self.extract_user_id(query)
    #     recommendations = self.recommend_items_for_user(user_id)
    #     return model_name, "Recommended items: " + ', '.join(map(str, recommendations)), "No accuracy information"
    
    def handle_query(self, query, model_name, dataset_name, use_new):
        self.query = query
        self.dataset_name = dataset_name

        dataset_file_path = f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_files\\{dataset_name}.csv'
        model_file_path = f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}.h5'
        metrics_file_path = f"D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{model_name}_metrics.json"

        if use_new:
            # 使用提供的dataset文件路径来训练新模型
            self.set_filepath(dataset_file_path)
            self.load_data()
            self.prepare_data()
            self.build_model()
            self.train_model(self.train_data, self.val_data)
            generated_model_name = self.generate_model_name(query)
            model_profile = self.generate_model_profile(query, generated_model_name)
            dataset_profile = self.generate_dataset_profile()
            self.save_model(generated_model_name, model_profile, dataset_profile)
            self.model = load_model(f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\{generated_model_name}.h5')
            # load_data.upload_and_move_files()
            # load_database.upload_and_move_files()
            
        else:
            # 加载现有的数据和模型
            self.set_filepath(dataset_file_path)
            self.load_data()
            self.prepare_data()
            self.model = load_model(model_file_path)

            if os.path.exists(metrics_file_path):
                with open(metrics_file_path, 'r') as file:
                    self.performance_metrics = json.load(file)
                print(f"Performance metrics loaded: {self.performance_metrics}")
            else:
                self.performance_metrics = None
                print("Performance metrics file does not exist.")
        
        user_id = self.extract_user_id(query)
        recommendations = self.recommend_items_for_user(user_id)
        return model_name, "Recommended items: " + ', '.join(map(str, recommendations)), self.performance_metrics



    # def handle_query(self, query, intent, model_name):
    #     self.query = query

    #     # 询问是否使用新的数据路径
    #     use_new_path = input("Using rag_search/spotify_dataset.csv. Do you need to provide a new file path? (yes/no): ").lower() == 'yes'
    #     if use_new_path:
    #         new_path = input("Please enter the new file path within the 'rag_search/' directory: ")
    #         self.set_filepath('rag_search/' + new_path)
    #         self.load_data()
    #         self.prepare_data()
    #         generated_model_name = self.generate_model_name(query)
    #         self.build_model()
    #         self.train_model()
    #         model_profile = self.generate_model_profile(query, generated_model_name)
    #         self.save_model(generated_model_name, model_profile)
    #         model_filename = f"{generated_model_name}.h5"
    #         self.model = load_model('rag_search/model_files/' + model_filename)
    #         user_id = self.extract_user_id(query)
    #         recommendations = self.recommend_items_for_user(user_id)
    #         return generated_model_name, "Recommended items: " + ', '.join(map(str, recommendations)), "No accuracy information"
    #     else:
    #         self.set_filepath('rag_search/spotify_dataset.csv')
    #         self.load_data()
    #         self.prepare_data()
    #         user_id = self.extract_user_id(query)
    #         model_filename = f"{model_name}.h5"
    #         self.model = load_model('rag_search/model_files/' + model_filename)

    #         recommendations = self.recommend_items_for_user(user_id)
    #         return model_name, "Recommended items: " + ', '.join(map(str, recommendations)), "No accuracy information"


    
    def extract_user_id(self, query):
        # Prompt for extracting user ID from the query
        prompt = f"""
        Given the user's query: '{query}', please identify and extract the unique user ID. 
        The user ID might be a numeric ID, a hexadecimal string, or any form of unique identifier embedded in the query.
        Please return only the numeric user ID. For example, if the query content is "please recommend playlist based on use id 4407", return "4407".
        """

        # Sending the prompt to the LLM
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )

        user_id = response.choices[0].message.content.strip()
        match = re.search(r"\d+", user_id)
        if match:
            user_id = match.group(0)  # Extract only the first occurrence of numbers
            print(f"Extracted User ID: {user_id}")
            return user_id
        else:
            raise ValueError("No numeric user ID found in the response.")


    def suggest_columns_for_model(self, query):
        if not query:
            raise ValueError("Query must not be empty for suggesting columns.")
        prompt = (
            f"Given the columns {self.data.columns.tolist()} in a dataset and the user query '{query}', suggest the appropriate column names for user IDs and item IDs."
            f""
        )
        prompt = (
            f"Given the columns {self.data.columns.tolist()} in a dataset and a user has made a request related to a binary classification recommendation system."
            f"The user's request is: '{query}'. In a binary classification recommendation system, user IDs are typically linked with item IDs to predict user preferences, such as music artist recommendations or playlist suggestions based on user interactions."
            f"Based on the user's request, please suggest the most appropriate column names for:"
            f"1. User IDs: typically a column identifying unique users."
            f"2. Item IDs: a column identifying items that can be recommended such as artists, tracks, or playlists."
            f"Please provide the column names for User IDs and Item IDs, separated by a comma without any spaces or quotation marks. For example, if the request is to recommend more musicians based on user names, respond with user_id,artistname. If the request is to recommend playlists based on user names, respond with user_id,playlistname."
        )
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        answer = response.choices[0].message.content.strip().split(',')
        print(answer)
        return answer
    
    def generate_model_name(self, query):
        # Prompt for generating a model name
        prompt = f"""
        Given the dataset with columns {self.data.columns.tolist()} and a recommendation system using the columns '{self.user_col}' for user IDs and '{self.item_col}' for item IDs, alongside the user's request: '{query}':
        Generate a unique and descriptive model name that reflects the purpose and functionality of the model. This name should be suitable for identifying the model's files and its profile.
        Please provide the model name in a concise format, suitable for filenames, without spaces or special characters.
        For example, if you want to name a model as useridplaylistrecommender, do not answer "modelname: useridplaylistrecommender", only answer "useridplaylistrecommender"
        """

        # Sending the prompt to the LLM
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        model_name = response.choices[0].message.content.strip().replace(' ', '_').replace(',', '').lower()
        print(f"Suggested model name: {model_name}")
        return model_name
    
    def generate_model_profile(self, query, model_name):
        # Setting up the prompt for the LLM to generate the model profile
        prompt = f"""
        Create a detailed model profile for '{model_name}' based on the following specifications:

        Model Name:
        {model_name}

        Model Overview:
        A binary classification recommendation system designed to suggest items (like artists or products) based on user interactions. The system uses columns '{self.user_col}' and '{self.item_col}' from a dataset to train a model aimed at fulfilling the user's specific request: '{query}'.

        Intended Use:
        This model is intended to be used in environments where personalization and user preference prediction are critical, for example in e-commerce or entertainment platforms, to enhance user experience by accurately predicting and recommending items.

        Technical Details:
        Algorithm Type: Mixed Collaborative Filtering with Neural Networks
        Input Features: User IDs and Item IDs from columns '{self.user_col}' and '{self.item_col}'
        Output: Probability scores indicating user preference

        Model Performance:
        - Accuracy: {self.performance_metrics['accuracy']}
        - Precision: {self.performance_metrics['precision']}
        - Recall: {self.performance_metrics['recall']}

        Limitations:
        Performance may degrade with sparse user-item interactions or limited diversity in the training data set.

        Please format the profile to be clear, professional, and detailed.
        """

        # Sending the prompt to the LLM
        response = self.ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
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
        Create a detailed dataset profile based on the following specifications:

        Dataset Name: {dataset_name}

        Overview:
        This dataset contains data structured in several columns: {column_names}. Below is a sample of the data to provide insight into the typical content and structure of the dataset:

        {sample_data}

        Usage:
        This dataset is primarily used for building recommendation models in the {dataset_name.split('_')[0]} domain. 

        Please format the profile to be clear, professional, and detailed.
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
    
    def choose_columns(self):
        # Attempt to use AI to suggest column names based on the user query
        user_query = self.query
        suggested_columns = self.suggest_columns_for_model(user_query)

        # Automatically use the AI's suggested columns if they are valid
        if suggested_columns and suggested_columns[0] in self.data.columns and suggested_columns[1] in self.data.columns:
            print(f"Using suggested columns based on your query: User ID -> {suggested_columns[0]}, Item ID -> {suggested_columns[1]}")
            self.user_col, self.item_col = suggested_columns
            return self.user_col, self.item_col
        else:
            print("AI's suggestion was invalid or not applicable.")
            print("Available columns in the dataset: ", self.data.columns.tolist())
            # If the suggestion is not valid, you might want to handle it differently or raise an error
            raise ValueError("AI suggested columns are invalid. Check the model and dataset compatibility.")


    
    # def choose_columns(self):

    #     while True:
    #         if self.user_col is not None and self.item_col is not None:
    #             return self.user_col, self.item_col

    #         # Use AI to suggest column names based on a user query
    #         user_query = self.query
    #         suggested_columns = self.suggest_columns_for_model(user_query)

    #         # Verify AI's suggested columns
    #         if suggested_columns[0] in self.data.columns and suggested_columns[1] in self.data.columns:
    #             print(f"Suggested columns based on your query: User ID -> {suggested_columns[0]}, Item ID -> {suggested_columns[1]}")
    #             if input("Would you like to use these columns? (yes/no): ").lower() == 'yes':
    #                 self.user_col, self.item_col = suggested_columns
    #                 return self.user_col, self.item_col
    #             else:
    #                 print("Please enter the column names manually.")
    #         else:
    #             print("AI's suggestion was invalid.")

    #         # Manual entry if AI suggestion is not used or invalid
    #         print("Available columns in the dataset: ", self.data.columns.tolist())
    #         self.user_col = input("Please enter the column name for user IDs: ")
    #         self.item_col = input("Please enter the column name for item IDs: ")
    #         if self.user_col in self.data.columns and self.item_col in self.data.columns:
    #             return self.user_col, self.item_col  # Valid column names
    #         else:
    #             print("Error: One or both column names are invalid. Please try again.")
    #             print("Valid columns are: ", self.data.columns.tolist())
    #             self.user_col, self.item_col = None, None  # Reset and continue the loop
        
# latent_dim_mf,在矩阵分解中，每个用户和项目被表示为4维的向量,根据数据集大小和复杂程度调整
# latent_dim_mlp=32 表示多层感知器中使用的嵌入维度
# reg_mf 和 reg_mlp是正则化项的系数，用于控制模型的过拟合。在嵌入层加入L2正则化可以帮助模型在训练过程中保持权重的稳定，避免过于依赖少数几个数据点，从而提高模型的泛化能力。



    def prepare_data(self, sample_size=10000, test_size=0.1, random_state=42):
        self.choose_columns()

        sample_df = self.data.sample(n=sample_size, random_state=random_state)

        sample_df[self.user_col] = self.user_encoder.fit_transform(sample_df[self.user_col])
        sample_df[self.item_col] = self.user_encoder.fit_transform(sample_df[self.item_col])

        interactions = []
        user_item_set = set(zip(sample_df[self.user_col], sample_df[self.item_col]))

        for (u, i) in user_item_set:
            interactions.append([u, i, 1])
            for _ in range(4):
                negative_item = np.random.choice(sample_df[self.item_col].unique())
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(sample_df[self.item_col].unique())
                interactions.append([u, negative_item, 0])

        interaction_df = pd.DataFrame(interactions, columns=[self.user_col, self.item_col, 'interactions'])
        self.train_data, self.val_data = train_test_split(interaction_df, test_size=test_size, random_state=random_state)
        
        self.number_of_users = self.train_data[self.user_col].max() +1
        self.number_of_items = self.train_data[self.item_col].max() +1

    def build_model(self, latent_dim_mf=4, latent_dim_mlp=32, reg_mf=0.1, reg_mlp=0.1):
        # input layer
        user = keras.layers.Input(shape=(), dtype="int64", name="user_id")
        item = keras.layers.Input(shape=(), dtype="int64", name="item_id")

        # 矩阵分解嵌入层
        mf_user_embedding = keras.layers.Embedding(
            input_dim=self.number_of_users+1,
            output_dim=latent_dim_mf,
            embeddings_regularizer=l2(reg_mf),
            name="mf_user_embedding",
            embeddings_initializer="RandomNormal",
            input_length=1
        )(user)
        mf_item_embedding = keras.layers.Embedding(
            input_dim=self.number_of_items+1,
            output_dim=latent_dim_mf,
            embeddings_regularizer=l2(reg_mf),
            name="mf_item_embedding",
            embeddings_initializer="RandomNormal",
            input_length=1
        )(item)

        # MLP 嵌入层
        mlp_user_embedding = keras.layers.Embedding(
            input_dim=self.number_of_users+1,
            output_dim=latent_dim_mlp,
            embeddings_regularizer=l2(reg_mlp),
            name="mlp_user_embedding",
            embeddings_initializer="RandomNormal",
            input_length=1
        )(user)
        mlp_item_embedding = keras.layers.Embedding(
            input_dim=self.number_of_items+1,
            output_dim=latent_dim_mlp,
            embeddings_regularizer=l2(reg_mlp),
            name="mlp_item_embedding",
            embeddings_initializer="RandomNormal",
            input_length=1
        )(item)

        # 合并层
        mf_vector = keras.layers.Multiply()([keras.layers.Flatten()(mf_user_embedding), keras.layers.Flatten()(mf_item_embedding)])
        mlp_vector = keras.layers.Concatenate()([
            keras.layers.Flatten()(mlp_user_embedding),
            keras.layers.Flatten()(mlp_item_embedding)
        ])

        for units in [64, 32]:  # 示例：多层感知机深度
            mlp_vector = keras.layers.Dense(units, activation='relu')(mlp_vector)

        final_vector = keras.layers.Concatenate()([mf_vector, mlp_vector])
        output = keras.layers.Dense(1, activation='sigmoid')(final_vector)

        self.model = keras.Model(inputs=[user, item], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
        self.model.summary()
     
    def train_model(self, train_data, val_data, epochs=2, batch_size=32):
        train_data_x = [train_data[self.user_col].values, train_data[self.item_col].values]
        train_data_y = train_data['interactions'].values
        val_data_x = [val_data[self.user_col].values, val_data[self.item_col].values]
        val_data_y = val_data['interactions'].values

        self.history = self.model.fit(train_data_x, train_data_y, epochs=epochs, batch_size=batch_size, validation_data=(val_data_x, val_data_y))
        print("Model training complete.")
        self.performance_metrics = {
            'accuracy': self.history.history['val_accuracy'][-1],
            'precision': self.history.history['val_precision'][-1],
            'recall': self.history.history['val_recall'][-1]
        }

    def save_model(self, model_name, model_profile, dataset_profile):
        model_filename = f"{model_name}.h5"
        model_profile_filename = f"{model_name}_profile.txt"
        dataset_profile_filename = f"{self.dataset_name}.txt"
        metrics_filename = f"{model_name}_metrics.json"

        self.model.save(os.path.join(f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\', model_filename))
        print(f"Model saved as {model_filename}")

        with open(os.path.join(f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_profile_files', model_profile_filename), 'w') as file:
            file.write(model_profile)

        with open(os.path.join(f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_profile_files', dataset_profile_filename), 'w') as file:
            file.write(dataset_profile)

        with open(os.path.join(f'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\model_files\\', metrics_filename), 'w') as file:
            json.dump(self.performance_metrics, file)

        print(f"Model profile saved as {model_profile_filename}")

    def recommend_items_for_user(self, user_id, num_recommendations=10):
        # Ensure user_id is an integer
        try:
            user_id = int(user_id)
        except ValueError:
            raise ValueError("Invalid user ID: must be a number.")
        
        # Get a list of all item IDs
        all_item_ids = np.array(list(set(self.train_data[self.item_col])))

        # Predict probabilities for all items
        predictions = self.model.predict([np.array([user_id] * len(all_item_ids)), all_item_ids])
        predictions = predictions.flatten()

        # Get item IDs with the highest probability
        top_indices = predictions.argsort()[-num_recommendations:][::-1]
        top_item_ids = all_item_ids[top_indices]

        return top_item_ids


if __name__ == "__main__":
    recommender = Recommender()
    recommender.query = "please recommend playlist based on use id 4407"
    recommender.set_filepath('rag_search/trx_data.csv')
    recommender.load_data()
    recommender.prepare_data()
    recommender.build_model()
    recommender.train_model(recommender.train_data, recommender.val_data) 

    user_input = input("请输入用户ID: ")
    recommended_items = recommender.recommend_items_for_user(user_input, 10)
    print(f"为用户 {user_input} 推荐的前 {len(recommended_items)} 个商品ID是：{recommended_items}")



    # load_data.upload_and_move_files()




# please recommend e-commerce Trax platform product id based on customer id 7172
# please recommend playlist based on use id 4407