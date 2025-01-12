import os

import textract

from lightrag import LightRAG, QueryParam
from lightrag.llm import (gpt_4o_mini_complete, ollama_embedding,
                          ollama_model_complete, openai_complete_if_cache)
from lightrag.utils import EmbeddingFunc

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

    
async def tongyi_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        # "qwen-plus-1125",
        # "qwen-turbo-0919",
        "qwen-plus-1220",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs
    )

async def moonshot_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "moonshot-v1-8k",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
        **kwargs
    )


WORKING_DIR = "./haojing"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
#     # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
# )

rag = LightRAG(
    working_dir=WORKING_DIR,
    # llm_model_func=ollama_model_complete,
    llm_model_func=tongyi_model_complete,
    # llm_model_name="qwen2.5-coder:32b",
    # llm_model_name="qwen2.5:latest",
    llm_model_max_async=1,
    llm_model_max_token_size=129024,
    # llm_model_kwargs={"host": "http://223.3.84.194:8010", "options": {"num_ctx": 32768}},
    # llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    # llm_model_kwargs={"host": "http://192.168.1.111:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts,
            embed_model="nomic-embed-text",
            host="http://223.3.84.194:8010",
            # texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            # texts, embed_model="nomic-embed-text", host="http://192.168.1.111:11434"
        ),
    ),
)

# with open("./book.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())

text_content = textract.process("./爱尔眼科(修正目录版).pdf")
rag.insert(text_content.decode("utf-8"))

# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# # Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )


print(rag.query("本次收购医院的成⽴时间？", param=QueryParam(mode="local")))
