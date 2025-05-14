import openai
from openai import OpenAI
import time
import os


def call_chatgpt(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    
    client = OpenAI(
        api_key="sk-zQyTk2qgolkoolUf5NT4iCZMu7wAWslW1jiX1rYblFzOwD6l",
        base_url="https://www.dmxapi.com/v1"  # 添加中转节点
    )
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    # 为 o3-mini 模型设置更大的 max_tokens
    if model == 'o3-mini':
        max_tokens = 2048  # 增加 token 限制

    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))

            # 为 o3-mini 模型移除 top_p 参数
            if model == 'o3-mini':
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=requested_completions
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions
                )

            completions.extend([choice.message.content for choice in response.choices])
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.RateLimitError as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')