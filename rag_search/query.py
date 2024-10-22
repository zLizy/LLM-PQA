# 古早版本主干查询系统
import pymongo

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import openai

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from gradio.themes.base import Base

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

import pandas as pd

import key_param  

import re
from textblob import TextBlob

load_dotenv()

AIclient = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

client = pymongo.MongoClient("mongodb+srv://arslan:771944972@cluster0.qv2ymat.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.sample_analytics
collection = db.transactions

dbName = "models"
collectionName = "collection_of_model_description"
model_collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_key=key_param.OPENAI_API_KEY)

vectorStore = MongoDBAtlasVectorSearch(model_collection, embeddings)

def query_model(query):
    docs = vectorStore.similarity_search(query,K=1)  #perform an atlas vector search using lang chain's vector store, retrieve the most similar document based on the query vector
    as_output = docs[0].page_content #extract the page content from the top document in the list(most relevant info)

    # llm = OpenAI(openai_api_key=key_param.OPENAI_API_KEY, temperature = 0) 
    llm = ChatOpenAI(openai_api_key=key_param.OPENAI_API_KEY)
    retriever = vectorStore.as_retriever()                                     #use LLM to fetch documents that are relevant to the query
    qa = RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever)
    retriever_output = qa.invoke(query)

    return as_output, retriever_output   #as_output is the most similar document from the atlas vector search, retriever_output is generated by RAG

# def simplify_model_description(description):
#     """
#     简化模型描述，尝试只提取模型名称。
#     """
#     # 假设模型名称通常出现在描述的开始，并且以特定词汇结束
#     match = re.search(r"^(.+?) is a", description)
#     if match:
#         return match.group(1)  # 返回匹配的模型名称
#     # return "Unknown"
#     return "Linear regression"  # 如果没有找到匹配，返回"Unknown"

def simplify_model_description(description):
    """
    根据描述内容提取出机器学习模型的名称。
    """

    # 使用TextBlob进行简单的NLP处理
    blob = TextBlob(description)
    sentences = blob.sentences  # 将文本分割成句子

    # 定义一组可能的机器学习模型名称
    model_names = [
        "Linear regression",
        "Logistic regression",
        "Decision tree",
        "Random forest",
        "Support vector machine",
        "Neural network",
        "K-nearest neighbors",
        "Naive Bayes",
        "Gradient boosting",
        "Twitter-roBERTa-base",
        "DPT-Hybrid"
    ]

    # # 初始化变量来存储最相关的模型名称和相关性得分
    # most_relevant_model = None
    # highest_relevance_score = 0

    # # 遍历每个句子和每个模型名称，查找与查询最相关的模型
    # for sentence in sentences:
    #     for model_name in model_names:
    #         if model_name.lower() in sentence.lower():  # 检查模型名称是否出现在句子中
    #             relevance_score = sentence.sentiment.polarity  # 使用情感极性作为简单的相关性得分
    #             if relevance_score > highest_relevance_score:
    #                 most_relevant_model = model_name
    #                 highest_relevance_score = relevance_score
    
    # return most_relevant_model if most_relevant_model else "Unknown"

    # 遍历模型名称列表，查找首个在描述中出现的模型名称
    for model_name in model_names:
        if model_name.lower() in description.lower():
            return model_name  # 返回首个匹配的模型名称

    return "Unknown" 
    
    # 构造一个正则表达式来匹配模型名称
    # pattern = r"(" + "|".join(model_names) + ")"
    # match = re.search(pattern, description, re.IGNORECASE)  # 忽略大小写进行匹配
    
    # if match:
    #     return match.group(0)  # 返回匹配的模型名称
    # else:
    #     return "Unknown"  # 如果没有找到匹配，返回"Unknown"




data = []


# 意图判断函数
def detect_intent_with_openai(query):
    prompt = f"请根据以下用户查询内容判断意图，并给出回答。如果查询的意图使用机器学习预测，请回答'0'，在query有类似 <I want to know the user 51617's next transaction amount。>的内容都被视为需要预测。否则，请回答'1'。\n\n用户查询：'{query}'\n\n回答："
    response = AIclient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    answer = response.choices[0].message.content

    print(answer)

    return answer

def extract_user_id_from_query(query):
    # 根据实际情况可能需要调整正则表达式
    # 这个正则表达式尝试匹配查询中的任何数字序列
    matches = re.findall(r'\b\d+\b', query)
    # 假设用户ID是查询中唯一的或最后一个数字序列
    if matches:
        # 可以根据实际情况选择返回第一个匹配、最后一个匹配或特定条件下的匹配
        return int(matches[-1])  # 返回最后一个匹配项作为用户ID
    else:
        return None

def tradition_database_query(user_id):
    return db.transactions.find({'account_id': user_id}).sort('date', -1).limit(1).next().get('transactions', [{}])[0].get('amount', '未找到交易金额')

def machine_learning(user_id):

    specific_account_id = user_id

    for item in collection.find({"account_id": specific_account_id}):
        # 以下为数据处理和提取代码
        account_id = item['account_id']
        transaction_count = item['transaction_count']
        bucket_start_date = item['bucket_start_date']
        bucket_end_date = item['bucket_end_date']
        
        for transaction in item['transactions']:
            transaction_date = transaction['date']
            amount = transaction['amount']
            transaction_code = transaction['transaction_code']
            symbol = transaction['symbol']
            price = float(transaction['price'])
            total = float(transaction['total'])

            data.append({
                "account_id": account_id,
                "transaction_count": transaction_count,
                "bucket_start_date": bucket_start_date,
                "bucket_end_date": bucket_end_date,
                "transaction_date": transaction_date,
                "amount": amount,
                "transaction_code": transaction_code,
                "symbol": symbol,
                "price": price,
                "total": total
            })

    df = pd.DataFrame(data)


    # 假设我们只使用'price'作为特征进行预测
    X = df[['price']]  # 特征集
    y = df['amount']  # 目标变量

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化并训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 使用测试集进行预测并计算准确度信息
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)  # 计算R^2分数

    # 假设我们用最后一次交易的价格作为预测输入
    last_transaction_price = df.iloc[-1]['price']
    next_transaction_amount_pred = model.predict([[last_transaction_price]])

    answer = int(next_transaction_amount_pred[0])

    # 准确度信息
    accuracy_info = f"R^2 score: {r2:.2f}"
    
    # 返回预测结果和准确度信息
    return answer, accuracy_info

    # 假设我们用用户最后一次交易的价格作为下一次交易的价格预测输入
    # 这里简单地选择数据集中的最后一条记录的价格作为示例
    # last_transaction_price = df.iloc[-1]['price']

    # # 使用模型进行预测
    # next_transaction_amount_pred = model.predict([[last_transaction_price]])

    # answer = int(next_transaction_amount_pred[0])

    # # 输出预测结果
    # print(f"预测的下一次交易金额为: {next_transaction_amount_pred[0]:.2f}")

    # return answer


# def process_query(query):
#     intent = detect_intent_with_openai(query)
#     print("chatgpt api 返回的信息是", intent)
#     if intent == "0":
#         return machine_learning()
#     else:
#         return tradition_database_query()

def process_query(query):

    # 尝试从查询中提取用户ID
    user_id = extract_user_id_from_query(query)
    
    if user_id is None:
        # 如果没有找到用户ID，可能需要返回一个错误信息或进行其他处理
        return "Error", "Cannot find user ID in the query."




    intent = detect_intent_with_openai(query)
    if intent == "1":
        db_result = tradition_database_query(user_id)
        return "traditional database query", db_result, "N/A"
    else:
        _, retriever_output = query_model(query)  # 仅使用query_model返回的模型描述
        print("Retriever Output:", retriever_output)

        if isinstance(retriever_output, dict):
            model_description = retriever_output.get('result', '')
        else:
            model_description = retriever_output

        simplified_model_name = simplify_model_description(model_description)
        prediction_result, prediction_accuracy_info = machine_learning(user_id)
        # ml_result = machine_learning(user_id)
        return simplified_model_name, prediction_result, prediction_accuracy_info
    

with gr.Blocks(theme=Base(),title="Question Answering App") as UI:
    gr.Markdown(
        """
        #Question Answering App
        """)
    textbox = gr.Textbox(label="Enter your Query")
    with gr.Row():
        button = gr.Button("Submit",variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Model Name or Query Method")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Query Result or Prediction")
        output3 = gr.Textbox(lines=1, max_lines=10, label="Prediction Accuracy Info or N/A for traditional query")

    # button.click(process_and_simplify_query,textbox,outputs=[output1])
    button.click(process_query, inputs=textbox, outputs=[output1, output2, output3])
    

UI.launch()



