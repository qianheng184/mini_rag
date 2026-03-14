import os
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions

def load_single_pdf(file_path):
    """
    Load a single PDF file and return its text content.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

#先创建一个测试文件
sample_text ="""这是一个测试文件，用于验证RAG系统的功能。
RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的技术，能够在生成文本时利用外部知识库中的信息。
通过RAG系统，我们可以在生成文本的过程中检索相关的信息，从而提高生成文本的准确性和相关性。"""
with open("sample.pdf", "w") as f:
    f.write(sample_text)
  
#初始化向量数据库
client = chromadb.Client()
collection = client.create_collection(
    name="my_knowledge"
)
#第三步，文档分块与存储
chunks = sample_text.split("。")
for i, chunk in enumerate(chunks):
    if chunk.strip():  # 忽略空块
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )
print(f"已存储 {len(chunks)} 个文本块")

# ---------- 第4步：检索 ----------
def retrieve(query, n_results=1):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results['documents'][0]

# ---------- 第5步：生成回答（先模拟，后续接入LLM）---------
def generate_answer(query):
    context = retrieve(query)
    # 第一阶段先用简单拼接演示
    return f"根据资料：{context}\n回答：请手动查看以上资料"

# 测试
print(generate_answer("什么是rag？"))