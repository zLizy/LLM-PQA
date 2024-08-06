import pymongo
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

client = pymongo.MongoClient("mongodb+srv://arslan:771944972@cluster0.qv2ymat.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.sample_analytics
collection = db.transactions

hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

data = []

# for item in collection.find().limit(100):  # 示例中仅处理前100条数据
#     account_id = item['account_id']
#     transaction_count = item['transaction_count']
#     bucket_start_date = item['bucket_start_date']  
#     bucket_end_date = item['bucket_end_date']
    
#     for transaction in item['transactions']:
#         transaction_date = transaction['date']
#         amount = transaction['amount']
#         transaction_code = transaction['transaction_code']
#         symbol = transaction['symbol']
#         price = float(transaction['price'])
#         total = float(transaction['total'])
        
#         data.append({
#             "account_id": account_id,
#             "transaction_count": transaction_count,
#             "bucket_start_date": bucket_start_date,  
#             "bucket_end_date": bucket_end_date, 
#             "transaction_date": transaction_date,  
#             "amount": amount,
#             "transaction_code": transaction_code,
#             "symbol": symbol,
#             "price": price,
#             "total": total
#         })

# df = pd.DataFrame(data)



print(f"用户 joel58 的最后一次交易金额为: {db.transactions.find({'account_id': 51617}).sort('date', -1).limit(1).next().get('transactions', [{}])[0].get('amount', '未找到交易金额')}")







# # 假设这是你的查询
# last_transaction_cursor = db.transactions.find({"account_id": 51617}).sort("date", -1).limit(1)

# try:
#     # 获取Cursor中的第一个（也是唯一一个）文档
#     last_transaction_doc = next(last_transaction_cursor, None)
    
#     # 确保文档存在
#     if last_transaction_doc is not None:
#         # 从文档中提取交易记录列表
#         transactions = last_transaction_doc.get('transactions', [])
        
#         # 确保交易记录列表不为空
#         if transactions:
#             # 获取最新的交易记录的金额
#             last_transaction_amount = transactions[0]['amount']
#             print(f"用户 joel58 的最后一次交易金额为: {last_transaction_amount}")
#         else:
#             print("未找到任何交易记录。")
#     else:
#         print("未找到符合条件的文档。")
# except Exception as e:
#     print(f"处理查询时发生错误：{e}")





# specific_account_id = 51617

# for item in collection.find({"account_id": specific_account_id}):
#     # 以下为数据处理和提取代码
#     account_id = item['account_id']
#     transaction_count = item['transaction_count']
#     bucket_start_date = item['bucket_start_date']
#     bucket_end_date = item['bucket_end_date']
    
#     for transaction in item['transactions']:
#         transaction_date = transaction['date']
#         amount = transaction['amount']
#         transaction_code = transaction['transaction_code']
#         symbol = transaction['symbol']
#         price = float(transaction['price'])
#         total = float(transaction['total'])

#         data.append({
#             "account_id": account_id,
#             "transaction_count": transaction_count,
#             "bucket_start_date": bucket_start_date,
#             "bucket_end_date": bucket_end_date,
#             "transaction_date": transaction_date,
#             "amount": amount,
#             "transaction_code": transaction_code,
#             "symbol": symbol,
#             "price": price,
#             "total": total
#         })

# df = pd.DataFrame(data)


# # 假设我们只使用'price'作为特征进行预测
# X = df[['price']]  # 特征集
# y = df['amount']  # 目标变量

# # 初始化并训练模型
# model = LinearRegression()
# model.fit(X, y)

# # 假设我们用用户最后一次交易的价格作为下一次交易的价格预测输入
# # 这里简单地选择数据集中的最后一条记录的价格作为示例
# last_transaction_price = df.iloc[-1]['price']

# # 使用模型进行预测
# next_transaction_amount_pred = model.predict([[last_transaction_price]])

# # 输出预测结果
# print(f"预测的下一次交易金额为: {next_transaction_amount_pred[0]:.2f}")











# 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练模型
# model = LinearRegression()
# model.fit(X_train, y_train)

# 进行预测
# y_pred = model.predict(X_test)

# 计算预测结果的评价指标
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# 打印评价指标
# print("MAE (Mean Absolute Error):", mae)
# print("MSE (Mean Squared Error):", mse)
# print("RMSE (Root Mean Squared Error):", rmse)

# 打印实际和预测的比较
# for actual, predicted in zip(y_test, y_pred):
#     print(f"Actual: {actual}, Predicted: {predicted:.2f}")

































# def generate_embedding(text: str) -> list[float]:

#   response = requests.post(
#     embedding_url,
#     headers={"Authorization": f"Bearer {hf_token}"},
#     json={"inputs": text})

#   if response.status_code != 200:
#     raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

#   return response.json()

# for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#   doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#   collection.replace_one({'_id': doc['_id']}, doc)

# query = "imaginary characters from outer space at war"

# results = collection.aggregate([
#   {"$vectorSearch": {
#     "queryVector": generate_embedding(query),
#     "path": "plot_embedding_hf",
#     "numCandidates": 100,
#     "limit": 4,
#     "index": "PlotSemanticSearch",
#       }}
# ]);

# for document in results:
#     print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')

