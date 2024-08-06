import os
import openai
import pandas as pd

# 配置OpenAI API密钥
client = openai.OpenAI(api_key='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')

def generate_code(user_query, sample_data, selected_file_path):

    prompt = f"""
    User Query: {user_query}
    Dataset Example: {sample_data}
    Dataset File Path: {selected_file_path}

    Please generate a Python code snippet to read the given CSV file and preprocess the data based on the user's query. The query will contain two parts: one part with requirements for data preprocessing (e.g., "only consider house age less than 30", "from the past six months"), and another part with information for machine learning tasks (e.g., "predict real estate price with transaction date 2012.917, house age 32, distance to the nearest MRT station 84.87882, number of convenience stores 10, latitude 24.98298, longitude 121.54024", "please recommend product id based on customer id 7172").

    Ignore the machine learning task information and only handle the data preprocessing requirements.

    Here are some examples for clarification:
    1. If the query is "only consider female data from the dataset, predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children", only filter the dataset to include female data, as shown below:
    
    ```python
    import pandas as pd

    # Read the CSV file
    data = pd.read_csv('D:/Program Files/Code repositories/RAG/RAG/rag_search/database_files/insurance.csv')

    # Keep only female data
    processed_data = data[data['sex'] == 'female']
    ```

    2. If the query is "only consider house age less than 30, predict real estate price with transaction date 2012.917,  house age 32, distance to the nearest MRT station 84.87882, number of convenience stores 10, latitude 24.98298, longitude 121.54024", only filter the dataset to include houses less than 30 years old, as shown below:

    ```python
    import pandas as pd

    # Read the CSV file
    data = pd.read_csv('D:/Program Files/Code repositories/RAG/RAG/rag_search/database_files/Real_estate.csv')

    # Filter houses with age less than 30
    processed_data = data[(data['X2 house age'] < 30)
    ```
    Ensure that the file paths in the generated code use double backslashes (\\\\) for Windows compatibility, and store the preprocessed data in a variable named 'processed_data'.
    """

    # 发送提示信息给LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Please generate data preprocessing code based on the following information."},
            {"role": "user", "content": prompt}
        ]
    )

    code = response.choices[0].message.content.strip()

    if code.startswith("```python") and code.endswith("```"):
        code = code[9:-3].strip()

    code = code.replace('your_dataset.csv', selected_file_path.replace("\\", "\\\\"))

    print("Generated Code:\n", code)
    
    # 执行生成的代码并返回处理后的数据
    local_vars = {}
    exec(code, {"pd": pd}, local_vars)
    processed_data = local_vars.get("processed_data") 

    return processed_data

def should_preprocess(user_query):
    prompt = f"""
    The user query is: "{user_query}"

    Determine if this query requires data preprocessing. 
    The query will contain two parts: one part with requirements for data preprocessing (e.g., "only consider house age less than 30", "from the past six months"), and another part with information for machine learning tasks (e.g., "predict real estate price with transaction date 2012.917, house age 32, distance to the nearest MRT station 84.87882, number of convenience stores 10, latitude 24.98298, longitude 121.54024", "please recommend product id based on customer id 7172").

    If the query includes any data preprocessing requirements, respond with a simple "yes".
    If the query only contains machine learning tasks and does not require any data preprocessing, respond with a simple "no".

    Examples:
    1. Query: "only consider female data from the dataset, predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children"
       Response: "yes"
    2. Query: "predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children"
       Response: "no"

    Respond only with "yes" or "no".
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )

    decision = response.choices[0].message.content.strip().lower()
    print('是否需要数据预处理的判断是：', decision)
    positive_responses = ["yes"]
    return any(phrase in decision for phrase in positive_responses)






# # 数据集文件路径
# database_path = r"D:\Program Files\Code repositories\RAG\RAG\rag_search\database_files"
# output_path = r"D:\Program Files\Code repositories\RAG\RAG\rag_search\dataset"

# # 列出数据库文件夹中的所有CSV文件
# csv_files = [f for f in os.listdir(database_path) if f.endswith('.csv')]

# # 让用户选择CSV文件
# print("请选择一个CSV文件进行处理：")
# for i, file in enumerate(csv_files):
#     print(f"{i + 1}. {file}")

# file_index = int(input("输入文件编号: ")) - 1
# selected_file = csv_files[file_index]
# selected_file_path = os.path.join(database_path, selected_file)

# # 读取CSV文件的前几行信息
# df = pd.read_csv(selected_file_path)
# sample_data = df.head().to_dict()

# # 用户查询
# user_query = input("请输入您的查询: ")


# # 调用生成代码并执行
# processed_data = generate_code(user_query, sample_data, selected_file_path)

# # 将处理后的数据保存到新路径
# if processed_data is not None:
#     output_file_path = os.path.join(output_path, f"processed_{selected_file}")
#     processed_data.to_csv(output_file_path, index=False)
#     print(f"处理后的数据已保存到: {output_file_path}")
# else:
#     print("未生成处理后的数据。")
















