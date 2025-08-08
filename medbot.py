import streamlit as st
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

hf_token=os.environ.get("HF_TOKEN")
DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_data
def get_vectorstore():
    embedding_model=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

def load_llm(huggingface_repo_id,hf_token):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":hf_token,
                      "max_length":"512"}
    )
    return llm




def main():
    st.title("ASK CHATBOT")
    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        custom_prompt_template="""
           use the piece of information provided in the context to answer user's question.
           If you don't know the answer, just say that you don't know, don't try to makeup the answer,
           dont provide anything out of the given context

           context:{context}
           question"{question}

           start the answer directly no small talk please
            """        
        
        huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"
         

        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                  llm=load_llm(huggingface_repo_id,hf_token=hf_token),
                  chain_type="stuff",
                  retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                  return_source_documents=True,
                  chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
                      )   
            response=qa_chain.invoke({'query':prompt})
            result=response["result"]
            source_document=response["source_document"]
            result_to_show=result+str(source_document)



        
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant','content':result_to_show})

        except Exception as e:
            st.error(f"error: {str(e)}")

if __name__=='__main__':
    main()        
