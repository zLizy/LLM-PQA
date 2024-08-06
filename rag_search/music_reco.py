# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from keras.models import load_model
# import pickle

# class ArtistRecommender:
#     def __init__(self):
#         self.model_path = 'spotify_recommendation_model.h5'
#         self.user_encoder_path = 'user_encoder.pickle'
#         self.artist_encoder_path = 'artist_encoder.pickle'
#         self.model = None
#         self.user_encoder = None
#         self.artist_encoder = None
#         self.check_and_load_model()

#     def check_and_load_model(self):
#         if os.path.exists(self.model_path) and os.path.exists(self.user_encoder_path) and os.path.exists(self.artist_encoder_path):
#             print("Loading existing model and encoders...")
#             self.model = load_model(self.model_path)
#             with open(self.user_encoder_path, 'rb') as f:
#                 self.user_encoder = pickle.load(f)
#             with open(self.artist_encoder_path, 'rb') as f:
#                 self.artist_encoder = pickle.load(f)
#         else:
#             print("Model and encoders do not exist, starting model training...")
#             self.train_model()

#     def prepare_data(self, df):
#         df = df.copy()
#         user_encoder = LabelEncoder()
#         artist_encoder = LabelEncoder()

#         df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
#         df['artist_id_encoded'] = artist_encoder.fit_transform(df['artistname'])
#         df['interaction'] = np.ones(len(df), dtype=int)

#         return df, user_encoder, artist_encoder
    
#     def handle_query(self, query, intent, model_name):
#         if intent == "1":
#             return "traditional database query", "TO DO", "N/A"
#         elif intent == "0" and model_name == "Artist Recommendation Model":
#             recommended_artists = self.recommend_artists(query)
#             if len(recommended_artists) > 0:
#                 recommended_artists_str = ', '.join(recommended_artists)
#                 return model_name, recommended_artists_str, "N/A"
#             else:
#                 return "Error", "Artist not found.", "N/A"
#         else:
#             return "Error", "Artist not found or no recommendations available.", "N/A"


#     def train_model(self):
#         spotify_df = pd.read_csv('rag_search/spotify_dataset.csv', skiprows=1, names=['user_id', 'artistname', 'trackname', 'playlistname'], on_bad_lines='skip')
#         train_df, test_df = train_test_split(spotify_df, test_size=0.2, random_state=42)
#         train_data, user_encoder, artist_encoder = self.prepare_data(train_df)
#         test_data, _, _ = self.prepare_data(test_df)

#         num_users = len(user_encoder.classes_)
#         num_artists = len(artist_encoder.classes_)
#         embedding_size = 50

#         user_input = keras.layers.Input(shape=(1,))
#         artist_input = keras.layers.Input(shape=(1,))
#         user_embedding = keras.layers.Embedding(num_users, embedding_size)(user_input)
#         artist_embedding = keras.layers.Embedding(num_artists, embedding_size)(artist_input)
#         merged = keras.layers.Dot(axes=2)([user_embedding, artist_embedding])
#         flattened = keras.layers.Flatten()(merged)
#         output = keras.layers.Dense(1, activation='sigmoid')(flattened)
#         model = keras.Model(inputs=[user_input, artist_input], outputs=output)
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#         model.fit([train_data['user_id_encoded'], train_data['artist_id_encoded']], train_data['interaction'],
#                   validation_data=([test_data['user_id_encoded'], test_data['artist_id_encoded']], test_data['interaction']),
#                   epochs=10, batch_size=64, callbacks=[early_stopping], verbose=1)
#         model.save(self.model_path)
#         with open(self.user_encoder_path, 'wb') as f:
#             pickle.dump(user_encoder, f)
#         with open(self.artist_encoder_path, 'wb') as f:
#             pickle.dump(artist_encoder, f)
#         self.model = model
#         self.user_encoder = user_encoder
#         self.artist_encoder = artist_encoder

#     def recommend_artists(self, artist_name, top_n=10):
#         if artist_name in self.artist_encoder.classes_:
#             artist_id = self.artist_encoder.transform([artist_name])[0]
#             artist_ids = np.arange(len(self.artist_encoder.classes_))
#             predictions = self.model.predict([np.array([artist_id] * len(artist_ids)), artist_ids])
#             recommended_ids = np.argsort(-predictions.flatten())[:top_n + 1]
#             recommended_ids = [id for id in recommended_ids if id != artist_id][:top_n]
#             return self.artist_encoder.inverse_transform(recommended_ids)
#         else:
#             return ["Artist not found: {}".format(artist_name)]

# if __name__ == "__main__":
#     recommender = ArtistRecommender()
#     artist_name = input("Enter artist name: ")
#     recommended_artists = recommender.recommend_artists(artist_name)
#     print("Recommended artists based on {}: ".format(artist_name))
#     for artist in recommended_artists:
#         print(artist)


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model
import pickle

class ArtistRecommender:
    def __init__(self):
        self.model_path = 'spotify_recommendation_model.h5'
        self.user_encoder_path = 'user_encoder.pickle'
        self.artist_encoder_path = 'artist_encoder.pickle'
        self.accuracy_info_path = 'spotify_recommendation_model_accuracy.pkl'
        self.model = None
        self.user_encoder = None
        self.artist_encoder = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.check_and_load_model()

    def check_and_load_model(self):
        if all(os.path.exists(path) for path in [self.model_path, self.user_encoder_path, self.artist_encoder_path, self.accuracy_info_path]):
            print("Loading existing model, encoders, and accuracy...")
            self.model = load_model(self.model_path)
            with open(self.user_encoder_path, 'rb') as f:
                self.user_encoder = pickle.load(f)
            with open(self.artist_encoder_path, 'rb') as f:
                self.artist_encoder = pickle.load(f)
            with open(self.accuracy_info_path, 'rb') as f:
                self.train_accuracy, self.val_accuracy = pickle.load(f)
        else:
            print("Model, encoders, or accuracy data do not exist, starting model training...")
            self.train_model()

    def prepare_data(self, df):
        df = df.copy()
        user_encoder = LabelEncoder()
        artist_encoder = LabelEncoder()

        df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
        df['artist_id_encoded'] = artist_encoder.fit_transform(df['artistname'])
        df['interaction'] = np.ones(len(df), dtype=int)

        return df, user_encoder, artist_encoder

    def train_model(self):
        spotify_df = pd.read_csv('rag_search/spotify_dataset.csv', skiprows=1, names=['user_id', 'artistname', 'trackname', 'playlistname'], on_bad_lines='skip')
        # train_df, test_df = train_test_split(spotify_df, test_size=0.2, random_state=42)
        # train_data, user_encoder, artist_encoder = self.prepare_data(train_df)
        # test_data, _, _ = self.prepare_data(test_df)

        sample_size = 100000
        sample_df = spotify_df.sample(n=sample_size, random_state=42)

        train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42)
        train_data, user_encoder, artist_encoder = self.prepare_data(train_df)
        test_data, _, _ = self.prepare_data(test_df)

        num_users = len(user_encoder.classes_)
        num_artists = len(artist_encoder.classes_)
        embedding_size = 50

        user_input = keras.layers.Input(shape=(1,))
        artist_input = keras.layers.Input(shape=(1,))
        user_embedding = keras.layers.Embedding(num_users, embedding_size)(user_input)
        artist_embedding = keras.layers.Embedding(num_artists, embedding_size)(artist_input)
        merged = keras.layers.Dot(axes=2)([user_embedding, artist_embedding])
        flattened = keras.layers.Flatten()(merged)
        output = keras.layers.Dense(1, activation='sigmoid')(flattened)
        model = keras.Model(inputs=[user_input, artist_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit([train_data['user_id_encoded'], train_data['artist_id_encoded']], train_data['interaction'],
                            validation_data=([test_data['user_id_encoded'], test_data['artist_id_encoded']], test_data['interaction']),
                            epochs=1, batch_size=64, callbacks=[early_stopping], verbose=1)

        self.train_accuracy = max(history.history['accuracy'])
        self.val_accuracy = max(history.history['val_accuracy'])

        # Save model, encoders, and accuracy
        model.save(self.model_path)
        with open(self.user_encoder_path, 'wb') as f:
            pickle.dump(user_encoder, f)
        with open(self.artist_encoder_path, 'wb') as f:
            pickle.dump(artist_encoder, f)
        with open(self.accuracy_info_path, 'wb') as f:
            pickle.dump((self.train_accuracy, self.val_accuracy), f)

        self.model = model
        self.user_encoder = user_encoder
        self.artist_encoder = artist_encoder
        print("Model trained and saved successfully.")

    def recommend_artists(self, artist_name, top_n=10):
        if artist_name in self.artist_encoder.classes_:
            artist_id = self.artist_encoder.transform([artist_name])[0]
            artist_ids = np.arange(len(self.artist_encoder.classes_))
            predictions = self.model.predict([np.array([artist_id] * len(artist_ids)), artist_ids])
            recommended_ids = np.argsort(-predictions.flatten())[:top_n + 1]
            recommended_ids = [id for id in recommended_ids if id != artist_id][:top_n]
            return self.artist_encoder.inverse_transform(recommended_ids)
        else:
            return ["Artist not found: {}".format(artist_name)]

    def handle_query(self, query, intent, model_name):
        if intent == "1":
            return "traditional database query", "TO DO", "N/A"
        elif intent == "0" and model_name == "Artist Recommendation Model":
            recommended_artists = self.recommend_artists(query)
            if len(recommended_artists) > 0:
                recommended_artists_str = ', '.join(recommended_artists)
                accuracy_info = f"Training Accuracy: {self.train_accuracy:.2f}, Validation Accuracy: {self.val_accuracy:.2f}"
                return model_name, recommended_artists_str, accuracy_info
            else:
                return "Error", "Artist not found.", "N/A"
        else:
            return "Error", "Artist not found or no recommendations available.", "N/A"

if __name__ == "__main__":
    recommender = ArtistRecommender()
    artist_name = input("Enter artist name: ")
    recommended_artists = recommender.recommend_artists(artist_name)
    print("Recommended artists based on {}: ".format(artist_name))
    for artist in recommended_artists:
        print(artist)
