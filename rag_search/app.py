from openai import OpenAI
import streamlit as st
from streamlit import spinner
from dotenv import load_dotenv
import os
import shelve
import re
import pandas as pd

from PIL import Image
import io
import base64

from query_test import initial_query,handle_new_model_selection,finalize_decision,beautify_final_response
from RAG import generate_code, should_preprocess

load_dotenv()

st.title("Question Answering Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

client = OpenAI(api_key='YOUR_API_KEY')

# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


# Load chat history from shelve file
def load_chat_history():
    # with shelve.open("chat_history") as db:
    #     return db.get("messages", [])
    try:
        with shelve.open("chat_history") as db:
            return db.get("messages", [])
    except (EOFError, KeyError):
        return []


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/image.py", label="Page 1", icon="1️⃣")      
    # st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
    st.page_link("http://www.google.com", label="Google", icon="🌎")
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
    
def process_user_input(prompt):
    # Use LLM to interpret the user's input
    decision = interpret_user_input(prompt)

    # models = ['RegressionModel', 'ClassificationModel', 'RecommendationModel']
    models = {
        'RegressionModel': ['Linear Regression', 'Random Forest', 'XGBoost'],
        'ClassificationModel': ['Random Forest', 'Neural Network'],
        'RecommendationModel': ['Neural Network']
    }
    datasets = os.listdir('D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_files')

    if "confirm" in decision.lower():
        # Extract additional required info if necessary
        return finalize_decision(st.session_state.initial_query, 'y', st.session_state.query_state)
    elif "change" in decision.lower():

        # st.write(st.session_state)

        model_name = st.session_state.query_state['model_name']
        matched_model_type = None

        if 'linear_regression' in model_name.lower():
            matched_model_type = 'RegressionModel'
        elif 'recommender' in model_name.lower():
            matched_model_type = 'RecommendationModel'
        elif 'classification' in model_name.lower():
            matched_model_type = 'ClassificationModel'
        # demo purpose    
        else:
            matched_model_type = 'RegressionModel'    

        for model_type, model_list in models.items():
            if model_name in model_list:
                matched_model_type = model_type
                break

        if matched_model_type:
            models_list = ", ".join(models[matched_model_type])
            example_model = models[matched_model_type][0]
            raw_response = (f"Based on the matched information, your query is suitable for the following **{matched_model_type}s**:\n\n"
                        f"**{models_list}**\n\n"
                        f"Please type the model you would like to use. For example, **'{example_model}'**.")
        else:
            raw_response = "Error: No matching model type found."

        # response = beautify_final_response(raw_response)
        
        return raw_response

        # models_list = "\n".join(f"**{model}**" for model in models)
        # models_list = "\n\n".join(f"**{category}:** " + ", ".join(models[category]) for category in models)
        # datasets_list = "\n".join(f"**{dataset}**" for dataset in datasets)
        # response = ("Please select a model and dataset from the lists below:\n\n"
        #             f"{models_list}\n\n"
        #             "Datasets:\n"
        #             f"{datasets_list}\n\n"
        #             "Please type the model and dataset you would like to use. For example, 'Linear Regression, insurance.csv'.")  
        # return response    
        # return f'Please select a model and dataset:, Models: **{", ".join(models)}** Datasets: **{", ".join(datasets)}**'
    elif "selection" in decision.lower():
        # Extract model and dataset choice from the decision or subsequent user input
        model_choice = extract_model(prompt)  # Define how to extract these

        user_query = st.session_state.initial_query
        dataset_choice = st.session_state.query_state['dataset_name'] + ".csv"
        if should_preprocess(user_query):
            selected_file_path = os.path.join('D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_files', dataset_choice)
            df = pd.read_csv(selected_file_path, on_bad_lines='skip')
            sample_data = df.head().to_dict()
            processed_data = generate_code(user_query, sample_data, selected_file_path)
            if processed_data is not None:
                output_file_path = os.path.join('D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_files', f"processed_{dataset_choice}")
                processed_data.to_csv(output_file_path, index=False)
                st.session_state['dataset_name'] = f"processed_{dataset_choice}"
                sample_data_markdown = processed_data.head().to_markdown(index=False)
                # response = f"We have preprocessed the data based on your query. The first few rows of the processed data are:\n\n{sample_data_markdown}\n\n"
                # response += handle_new_model_selection(model_choice, f"processed_{dataset_choice}", st.session_state.initial_query)
                response = handle_new_model_selection(model_choice, f"processed_{dataset_choice}", st.session_state.initial_query)
            else:
                response = "Data preprocessing failed."
        else:
            response = handle_new_model_selection(model_choice, dataset_choice, st.session_state.initial_query)
        return response

        # return handle_new_model_selection(model_choice, dataset_choice, st.session_state.initial_query)
    elif "query" in decision.lower():
        st.session_state.initial_query = prompt
        response, state = initial_query(prompt)
        st.session_state.query_state = state
        return response
    elif "guide" in decision.lower():
        # Handle non-specific or chat-like interactions
        # response = 'To query, input your question directly. If dissatisfied with the provided model and dataset, you can upload a new dataset file to the system and specify the file name and desired machine learning model type.'
        response = ('To query, input your question directly. '
                    'You can include specific feature values in your question to get more accurate results. '
                    'For example, **"predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children"**. '
                    'If dissatisfied with the provided model and dataset, you can specify the dataset file name and desired machine learning model type. '
                    'To use a new dataset, please place the file in the local "database_files" folder.')    
        return response  
    else:
        full_response = ""
        # Format the prompt for OpenAI API
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
        return full_response

def interpret_user_input(prompt):
        
    prompt = f"""
    Based on the user's input, categorize and direct the action as follows:

    - If the input is a direct query like:
        "predict insurance charge for a 19 year old female, non-smoker, living in northeast with a BMI of 27.9 and no children"
        "predict real estate price with transaction date 2012.917, house age 32, distance to the nearest MRT station 84.87882, number of convenience stores 10, latitude 24.98298, longitude 121.54024"
        "please recommend playlist based on user id 4407"
       Respond as "query".

    - If the input is an affirmative response like:
        "y"
        "yes"
        "I want to use matched model and dataset"
      Respond as "confirm".

    - If the input suggests a desire for a new model, like:
        "new"
        "I want to use new model"
        "Can I select another model?"
        "I want to train a new model"
      Respond with "change". 
      
    - If the input specifies a choice for a new model like:
        "ClassificationModel"
        "I want to use model RegressionModel"
      Respond with "selection".

    - For input requesting help or instructions like:
        "how to use this system"
        "help"
        "user guide"
      Respond with "guide"

    - For input requesting uploading an image like:
        "image"
        "classify an image"
      Respond with "image"

    - For any other type of input
      Respond with "chat"

    User input: "{prompt}"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250  # Adjust based on the complexity of your prompt and expected responses
    )

    # Extracting the LLM's decision
    decision = response.choices[0].message.content.strip()
    print('UI输入的判断是:', decision)
    return decision

def extract_model(prompt):
        
    prompt = f"""
    Based on the user's input, extract the model name. For example:
    If the user's input is "I want to use model insurancecharge_regression",
    respond with 'modelname: insurancecharge_regression'.
    If the user's input is "real_estate_regression",
    respond with 'modelname: real_estate_regression'.    
    User input: "{prompt}"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    extracted_text = response.choices[0].message.content.strip()

    # 使用正则表达式从LLM回复中提取模型名和数据集名
    model_match = re.search(r"modelname:\s*(\S+)", extracted_text, re.IGNORECASE)

    # 检查是否成功提取到模型名和数据集名
    if model_match:
        model_name = model_match.group(1)
        print(f'提取的模型名为: {model_name}')
        return model_name
    else:
        # 如果没有提取到，可以返回None或者进一步处理
        print('未能提取到模型名')
        return None, None
    
def is_base64_image(data):
    if isinstance(data, str) and data.startswith("data:image/"):
        return True
    return False

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with spinner("Processing your request, please wait..."):
            message_placeholder = st.empty()
            full_response = ""

            LLM_response = process_user_input(prompt)

            for response in LLM_response:
                    # full_response += response.get('content', '')
                full_response += response
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
    
# Main chat interface
# if prompt := st.chat_input("How can I help?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user", avatar=USER_AVATAR):
#         st.markdown(prompt)

#     with st.chat_message("assistant", avatar=BOT_AVATAR):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=st.session_state["messages"],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.content or ""
#             message_placeholder.markdown(full_response + "|")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
























# # Ensure openai_model is initialized in session state
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# # Load chat history from shelve file
# def load_chat_history():
#     with shelve.open("chat_history") as db:
#         return db.get("messages", [])
    
# # Save chat history to shelve file
# def save_chat_history():
#     with shelve.open("chat_history") as db:
#         db["messages"] = st.session_state.messages

# def remove_duplicates(messages):
#     seen = set()
#     unique_messages = []
#     for message in messages:
#         content = message['content']
#         if content not in seen:
#             seen.add(content)
#             unique_messages.append(message)
#     return unique_messages

# def display_messages():
#     unique_messages = remove_duplicates(st.session_state.messages)
#     for message in unique_messages:
#         avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
#         with st.chat_message(message["role"], avatar=avatar):
#             st.markdown(message["content"])
    
# # Initialize or load chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = load_chat_history()

# # Sidebar with a button to delete chat history
# with st.sidebar:
#     if st.button("Delete Chat History"):
#         st.session_state.messages = []
#         save_chat_history()

# display_messages()

# def get_dataset_files():
#     full_path = 'D:\\Program Files\\Code repositories\\RAG\\RAG\\rag_search\\database_files'
#     return [file for file in os.listdir(full_path) if file.endswith('.csv')]

# user_input = st.text_input("Enter your query or decision:", key="user_query_input")

# if user_input:

#     if 'initial_query_done' not in st.session_state:

#         st.session_state.messages.append({"role": "user", "content": user_input})
#         st.session_state.messages.append({"role": "assistant", "content": "Searching for matching models and datasets, please wait..."})
#         display_messages() 

#         response, state = initial_query(user_input)
#         st.session_state.initial_query_done = True
#         st.session_state.initial_query = user_input
#         st.session_state.query_response = response
#         st.session_state.query_state = state

#         st.session_state.messages.append({"role": "assistant", "content": response})
#         display_messages()

#     elif user_input.lower() in ['y', 'new']:
#         if user_input.lower() == 'new':
#             model_choice = st.selectbox("Select a model:", ['RegressionModel', 'Recommender'])
#             dataset_files = get_dataset_files()
#             dataset_choice = st.selectbox("Select a dataset:", dataset_files)
#             response = handle_new_model_selection(model_choice, dataset_choice, st.session_state.initial_query)
#         else:
#             response = finalize_decision(st.session_state.initial_query, user_input, st.session_state.query_state)

#         st.session_state.initial_query_done = None  # Reset query response state
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         display_messages()

# # Save chat history after each interaction
# save_chat_history()

# # Save chat history after each interaction
# def save_chat_history():
#     with shelve.open("chat_history") as db:
#         db["messages"] = st.session_state.messages

