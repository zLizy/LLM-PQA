import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model

# sales_history_df = pd.read_csv('rag_search/trx_data.csv', skiprows=1, names=['customerId','products'], on_bad_lines='skip')

spotify_df = pd.read_csv('rag_search/spotify_dataset.csv', skiprows=1, names=['user_id', 'artistname', 'trackname', 'playlistname'], on_bad_lines='skip')
print(spotify_df.head())


sample_df = spotify_df.sample(n=10000, random_state=42)



# # example 1: split product items
# sales_history_df['products'] = sales_history_df['products'].apply(lambda x: [int(i) for i in x.split('|')])
# print(sales_history_df.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index())

# user_interacted_product = pd.melt(sales_history_df.set_index('customerId')['products'].apply(pd.Series).reset_index(), 
#                                      id_vars=['customerId'],
#                                      value_name='products' 
#                                     ).dropna().drop(['variable'], axis=1).rename(columns={'products': 'productId'}).reset_index(drop=True)
# user_interacted_product['productId'] = user_interacted_product['productId'].astype(np.int64)
# print(user_interacted_product.head())

user_encoder = LabelEncoder()
artist_encoder = LabelEncoder()

sample_df['user_id'] = user_encoder.fit_transform(sample_df['user_id'])
sample_df['artistname'] = artist_encoder.fit_transform(sample_df['artistname'])

# data_users = user_interacted_product['customerId']
# data_items = user_interacted_product['productId']

# # split the data test and train
# train_users, test_users, train_items, test_items = train_test_split(data_users, data_items, 
#                                                                     test_size=0.01, random_state=42, shuffle=True)
# train_data = pd.DataFrame((zip(train_users, train_items)),columns=['customerId', 'productId'])

train_users, test_users, train_artists, test_artists = train_test_split(sample_df['user_id'], sample_df['artistname'], 
                                                                        test_size=0.1, random_state=42, shuffle=True)
train_data = pd.DataFrame({'user_id': train_users, 'artistname': train_artists})
train_data['interactions'] = 1  # 假设所有选定的用户-艺术家对都是正面交互
print(train_data.head())

print('————————————————————————————分割线——————————————————————————————')

# Get a list of all artist IDs
all_artist_ids = train_data['artistname'].unique()
all_user_ids = train_data['user_id'].unique()

# Placeholders that will hold the training data
user_ids, artist_ids, interactions = [], [], []

# This is the set of artists that each user has interacted with
user_artist_set = set(zip(train_data['user_id'], train_data['artistname']))

# 4:1 ratio of negative to positive samples
num_negatives = 4
for (u, a) in tqdm(user_artist_set):
    user_ids.append(u)
    artist_ids.append(a)
    interactions.append(1)  # Items that the user has interacted with are positive

    # Generating negative samples
    for _ in range(num_negatives):
        negative_artist = np.random.choice(all_artist_ids)
        # Ensure the user has not interacted with this artist
        while (u, negative_artist) in user_artist_set:
            negative_artist = np.random.choice(all_artist_ids)
        user_ids.append(u)
        artist_ids.append(negative_artist)
        interactions.append(0)  # Items not interacted with are negative

# Create a DataFrame for the interaction matrix
interaction_matrix = pd.DataFrame({'user_id': user_ids, 'artistname': artist_ids, 'interactions': interactions})

print(interaction_matrix.head())

data_x = np.array(interaction_matrix[['user_id', 'artistname']].values)
data_y = np.array(interaction_matrix[['interactions']].values)
# split validation data
train_data_x, val_data_x, train_data_y, val_data_y = train_test_split(data_x, data_y, test_size=0.1, random_state=42, shuffle=True)
print("Train Data Shape {}".format(train_data_x.shape))
print("Validation Data Shape {}".format(val_data_x.shape))

# train data
train_data_users = train_data_x[:,0]
train_data_items = train_data_x[:,1]
# validation data
val_data_users = val_data_x[:,0]
val_data_items = val_data_x[:,1]



number_of_users = train_data['user_id'].max()
number_of_items = train_data['artistname'].max()
latent_dim_mf = 4
latent_dim_mlp = 32
reg_mf = 0
reg_mlp = 0.1
dense_layers = [8, 4]
reg_layers = [0.1, 0.1]
activation_dense = "relu"



 # input layer
user = keras.layers.Input(shape=(), dtype="int64", name="user_id")
item = keras.layers.Input(shape=(), dtype="int64", name="item_id")

# 矩阵分解嵌入层
mf_user_embedding = keras.layers.Embedding(
    input_dim=number_of_users+1,
    output_dim=latent_dim_mf,
    name="mf_user_embedding",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    input_length=1,
    )
mf_item_embedding = keras.layers.Embedding(
    input_dim=number_of_items+1,
    output_dim=latent_dim_mf,
    name="mf_item_embedding",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mf),
    input_length=1,
    )
# 多层感知机嵌入层
mlp_user_embedding = keras.layers.Embedding(
    input_dim=number_of_users+1,
    output_dim=latent_dim_mlp,
    name="mlp_user_embedding",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mlp),
    input_length=1,
    )
mlp_item_embedding = keras.layers.Embedding(
    input_dim=number_of_items+1,
    output_dim=latent_dim_mlp,
    name="mlp_item_embedding",
    embeddings_initializer="RandomNormal",
    embeddings_regularizer=l2(reg_mlp),
    input_length=1,
    )

# MF vector
mf_user_latent = keras.layers.Flatten()(mf_user_embedding(user))
mf_item_latent = keras.layers.Flatten()(mf_item_embedding(item))
mf_cat_latent = keras.layers.Multiply()([mf_user_latent, mf_item_latent])

# MLP vector
mlp_user_latent = keras.layers.Flatten()(mlp_user_embedding(user))
mlp_item_latent = keras.layers.Flatten()(mlp_item_embedding(item))
mlp_cat_latent = keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])

mlp_vector = mlp_cat_latent

# build dense layers for model
for i in range(len(dense_layers)):
    layer = keras.layers.Dense(
            dense_layers[i],
            activity_regularizer=l2(reg_layers[i]),
            activation=activation_dense,
            name="layer%d" % i)
    mlp_vector = layer(mlp_vector)

NeuMf_layer = keras.layers.Concatenate()([mf_cat_latent, mlp_vector])

result = keras.layers.Dense(1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction")
output = result(NeuMf_layer)
model = keras.Model(inputs=[user, item], outputs=output)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ],
)


history = model.fit(x=[train_data_users, train_data_items], y=train_data_y, 
                      batch_size=16, epochs=2, 
                      validation_data=([val_data_users, val_data_items], val_data_y))

# 保存模型 TO DO，自动修改模型名，并且将模型名返回到数据库 
model.save('spotify_model.h5')


# loss = history.history['loss']
# val_loss = history.history['val_loss']
# # loss
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.title(f'Training and Validation Loss. \nTrain Loss: {str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
# plt.xlabel('epoch')
# plt.show()

# model = load_model('spotify_model.h5')

# def recommend_items_for_user(user_id, num_recommendations=10):
#     # 确保用户ID是整数
#     user_id = int(user_id)
    
#     # 获取所有商品的ID列表
#     all_product_ids = np.array(list(set(train_data['artistname'])))

#     # 预测所有商品的概率
#     predictions = model.predict([np.array([user_id]*len(all_product_ids)), all_product_ids])
#     predictions = predictions.flatten()

#     # 获取最高概率的商品ID
#     top_indices = predictions.argsort()[-num_recommendations:][::-1]
#     top_product_ids = all_product_ids[top_indices]
    
#     # 如果数据进行了编码转换，这里需要进行逆转换
#     # top_product_ids = products_encoder.inverse_transform(top_product_ids)

#     top_artist_names = artist_encoder.inverse_transform(top_product_ids)

#     return top_artist_names

# # 用户输入
# user_input = input("请输入用户ID: ")
# recommended_items = recommend_items_for_user(user_input)
# print(f"为用户 {user_input} 推荐的前 {len(recommended_items)} 个商品ID是：{recommended_items}")

# (User=>bought item) pairs for testing
# test_user_item_set = list(set(zip(test_users, test_artists)))

# # Dict of all items that are interacted with by each user
# user_interacted_items = sample_df.groupby('user_id')['artistname'].apply(list).to_dict()

# hits = []

# sample_size = int(len(test_user_item_set) * 0.5)

# for (u, i) in tqdm(random.sample(test_user_item_set, sample_size)):
#     interacted_items = user_interacted_items[u]
#     not_interacted_items = set(all_artist_ids) - set(interacted_items)
#     selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
#     test_items = selected_not_interacted + [i]
    
#     predicted_labels = np.squeeze(model.predict([np.array([u]*len(test_items)), np.array(test_items)]))
    
#     top10_items = [test_items[j] for j in np.argsort(predicted_labels)[::-1][:10]]
    
#     if i in top10_items:
#         hits.append(1)
#     else:
#         hits.append(0)
        
# print("The Hit Ratio @ 10 is {:.3f}".format(np.average(hits)))

# train_data['interactions'] = 1
# print(train_data.head())

# # Get a list of all sku_ids
# all_product_ids = train_data['productId'].unique()
# all_customer_ids = train_data['customerId'].unique()
# # Placeholders that will hold the training data
# customerId, productId, interactions = [], [], []
# # This is the set of items that each user has interaction with
# customer_product_set = set(zip(train_data['customerId'], train_data['productId']))
# # 4:1 ratio of negative to positive samples
# num_negatives = 4
# for (u, i) in tqdm(customer_product_set):
#     customerId.append(u)
#     productId.append(i)
#     interactions.append(1) # items that the user has interacted with are positive
#     for _ in range(num_negatives):
#         # randomly select an item
#         negative_item = np.random.choice(all_product_ids) 
#         # check that the user has not interacted with this item
#         while (u, negative_item) in customer_product_set:
#             negative_item = np.random.choice(all_product_ids)
#         customerId.append(u)
#         productId.append(negative_item)
#         interactions.append(0) # items not interacted with are negative
# interaction_matrix = pd.DataFrame(list(zip(customerId, productId, interactions)),columns=['customerId', 'productId', 'interactions'])

# print('————————————————————————————分割线——————————————————————————————')
# print(interaction_matrix.head())




# data_x = np.array(interaction_matrix[['customerId', 'productId']].values)
# data_y = np.array(interaction_matrix[['interactions']].values)
# # split validation data
# train_data_x, val_data_x, train_data_y, val_data_y = train_test_split(data_x, data_y, test_size=0.1, random_state=42, shuffle=True)
# print("Train Data Shape {}".format(train_data_x.shape))
# print("Validation Data Shape {}".format(val_data_x.shape))


# # train data
# train_data_users = train_data_x[:,0]
# train_data_items = train_data_x[:,1]
# # validation data
# val_data_users = val_data_x[:,0]
# val_data_items = val_data_x[:,1]



# number_of_users = train_data['customerId'].max()
# number_of_items = train_data['productId'].max()
# latent_dim_mf = 4
# latent_dim_mlp = 32
# reg_mf = 0
# reg_mlp = 0.1
# dense_layers = [8, 4]
# reg_layers = [0.1, 0.1]
# activation_dense = "relu"



#  # input layer
# user = keras.layers.Input(shape=(), dtype="int64", name="user_id")
# item = keras.layers.Input(shape=(), dtype="int64", name="item_id")

# # 矩阵分解嵌入层
# mf_user_embedding = keras.layers.Embedding(
#     input_dim=number_of_users+1,
#     output_dim=latent_dim_mf,
#     name="mf_user_embedding",
#     embeddings_initializer="RandomNormal",
#     embeddings_regularizer=l2(reg_mf),
#     input_length=1,
#     )
# mf_item_embedding = keras.layers.Embedding(
#     input_dim=number_of_items+1,
#     output_dim=latent_dim_mf,
#     name="mf_item_embedding",
#     embeddings_initializer="RandomNormal",
#     embeddings_regularizer=l2(reg_mf),
#     input_length=1,
#     )
# # 多层感知机嵌入层
# mlp_user_embedding = keras.layers.Embedding(
#     input_dim=number_of_users+1,
#     output_dim=latent_dim_mlp,
#     name="mlp_user_embedding",
#     embeddings_initializer="RandomNormal",
#     embeddings_regularizer=l2(reg_mlp),
#     input_length=1,
#     )
# mlp_item_embedding = keras.layers.Embedding(
#     input_dim=number_of_items+1,
#     output_dim=latent_dim_mlp,
#     name="mlp_item_embedding",
#     embeddings_initializer="RandomNormal",
#     embeddings_regularizer=l2(reg_mlp),
#     input_length=1,
#     )

# # MF vector
# mf_user_latent = keras.layers.Flatten()(mf_user_embedding(user))
# mf_item_latent = keras.layers.Flatten()(mf_item_embedding(item))
# mf_cat_latent = keras.layers.Multiply()([mf_user_latent, mf_item_latent])

# # MLP vector
# mlp_user_latent = keras.layers.Flatten()(mlp_user_embedding(user))
# mlp_item_latent = keras.layers.Flatten()(mlp_item_embedding(item))
# mlp_cat_latent = keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])

# mlp_vector = mlp_cat_latent

# # build dense layers for model
# for i in range(len(dense_layers)):
#     layer = keras.layers.Dense(
#             dense_layers[i],
#             activity_regularizer=l2(reg_layers[i]),
#             activation=activation_dense,
#             name="layer%d" % i)
#     mlp_vector = layer(mlp_vector)

# NeuMf_layer = keras.layers.Concatenate()([mf_cat_latent, mlp_vector])

# result = keras.layers.Dense(1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction")
# output = result(NeuMf_layer)
# model = keras.Model(inputs=[user, item], outputs=output)

# model.summary()




# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss="binary_crossentropy",
#     metrics=[
#         tf.keras.metrics.BinaryAccuracy(name="accuracy"),
#         tf.keras.metrics.Precision(name="precision"),
#         tf.keras.metrics.Recall(name="recall")
#     ],
# )


# history = model.fit(x=[train_data_users, train_data_items], y=train_data_y, 
#                       batch_size=16, epochs=5, 
#                       validation_data=([val_data_users, val_data_items], val_data_y))

# # 保存模型 TO DO，自动修改模型名，并且将模型名返回到数据库 
# model.save('trained_model.h5')


# loss = history.history['loss']
# val_loss = history.history['val_loss']
# # loss
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.title(f'Training and Validation Loss. \nTrain Loss: {str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
# plt.xlabel('epoch')
# plt.show()



# model = load_model('trained_model.h5')

# # # (User=>bought item) pairs for testing
# # test_user_item_set = list(set(zip(test_users, test_items)))

# # # Dict of all items that are interacted with by each user
# # user_interacted_items = user_interacted_product.groupby('customerId')['productId'].apply(list).to_dict()

# # hits = []

# # sample_size = int(len(test_user_item_set) * 0.5)

# # for (u, i) in tqdm(random.sample(test_user_item_set, sample_size)):
# #     interacted_items = user_interacted_items[u]
# #     not_interacted_items = set(all_product_ids) - set(interacted_items)
# #     selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
# #     test_items = selected_not_interacted + [i]
    
# #     predicted_labels = np.squeeze(model.predict([np.array([u]*len(test_items)), np.array(test_items)]))
    
# #     top10_items = [test_items[j] for j in np.argsort(predicted_labels)[::-1][:10]]
    
# #     if i in top10_items:
# #         hits.append(1)
# #     else:
# #         hits.append(0)
        
# # print("The Hit Ratio @ 10 is {:.3f}".format(np.average(hits)))


def recommend_items_for_user(user_id, num_recommendations=10):
    # 确保用户ID是整数
    user_id = int(user_id)
    
    # 获取所有商品的ID列表
    all_product_ids = np.array(list(set(train_data['artistname'])))

    # 预测所有商品的概率
    predictions = model.predict([np.array([user_id]*len(all_product_ids)), all_product_ids])
    predictions = predictions.flatten()

    # 获取最高概率的商品ID
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    top_product_ids = all_product_ids[top_indices]
    
    # 如果数据进行了编码转换，这里需要进行逆转换
    # top_product_ids = products_encoder.inverse_transform(top_product_ids)

    return top_product_ids

# 用户输入
user_input = input("请输入用户ID: ")
recommended_items = recommend_items_for_user(user_input)
print(f"为用户 {user_input} 推荐的前 {len(recommended_items)} 个商品ID是：{recommended_items}")