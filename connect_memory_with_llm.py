from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS

hf_token=os.environ.get("HF_TOKEN")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":hf_token,
                      "max_length":"512"}
    )
    return llm
DB_FAISS_PATH="vectorstore/db_faiss"
custom_prompt_template="""
use the piece of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to makeup the answer,
dont provide anything out of the given context

context:{context}
question"{question}

start the answer directly no small talk please
"""
def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
)
user_query=input("write query here")
response=qa_chain.invoke({'query':user_query})
print("RESULT", response["result"])
