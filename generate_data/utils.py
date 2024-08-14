import pandas as pd
import json
import os
import time
import random
from tqdm import tqdm
from openai import OpenAI
import openai


def encode_prompt_and_generate_data(rag_r, prompt_path, seed_path, model_name_id):

    client = OpenAI(base_url="http://10.54.10.127:9997/v1", api_key="sk-ns26vudyGLPMi")
    file_name = os.path.basename(prompt_path)
    file_name, _ = os.path.splitext(file_name)
    seed_data = []
    prompt = open(prompt_path, encoding="utf-8").read()
    with open(seed_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除前后空格
            if line:  # 忽略空行
                try:
                    seed_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e} for line: {line}")
                    continue
    sample_e = random.sample(seed_data, 1)

    for item in sample_e:
        rag = item["rag"]
        rag_toxic = item[f"{file_name}_rag_toxic"]
        prompt += "###\n"
        prompt += f"<context>\n{rag}\n</context>" + "\n"
        prompt += f"answer:" + '\n'
        prompt += f"{rag_toxic}" + "\n"

    prompt += "###\n"
    prompt += f"<context>\n{rag_r}\n</context>" + "\n"
    prompt += f"answer:" + '\n'

    try:
        completion_batch = client.chat.completions.create(
            model=model_name_id
            ,
            messages=[
                {"role": "system", "content": "You are a rewritten intelligent assistant！"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16000,
            stream=False,
            temperature=0.9,
            top_p=0.85
        )

        text = completion_batch.choices[0].message.content

        if text:
            rag_toxic = text.strip()
            return {f"{file_name}_rag_toxic": rag_toxic}
        else:
            return ''
    except Exception as e:
        print(e)


# def post_process_generate_data(content) -> str:
#     if content is not None:
#
#         rag_toxic = content.strip()
#         return rag_toxic
#
#     return ''
