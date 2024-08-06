

import requests
from bs4 import BeautifulSoup

def get_model_description_from_page(model_url):
    """
    从Hugging Face模型页面获取模型描述。
    """
    response = requests.get(model_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # 尝试找到包含描述的标签
        # 注意：以下选择器可能需要根据实际页面结构进行调整
        description_tag = soup.find('article')  # 假设描述在<article>标签内
        if description_tag:
            # 清理并返回文本内容
            return ' '.join(description_tag.text.split())
        else:
            return "Description not found."
    else:
        return "Failed to load the model page."

# 模型页面URL
model_url = "https://huggingface.co/ByteDance/SDXL-Lightning"

# 获取并打印模型描述
description = get_model_description_from_page(model_url)
print("Model Description:", description)
