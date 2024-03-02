import os
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb = Chroma(persist_directory="chroma.sqlite3", embedding_function=embeddings)

vectordb.persist()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True,)


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True
                                             )

pipe=pipeline("text-generation",
              model=model,
              tokenizer=tokenizer,
              torch_dtype=torch.bfloat16,
              device_map='auto',
              max_new_tokens=512,
              min_new_tokens=-1,
              top_k=30

              )


llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})

memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)

pdf_qa=ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=vectordb.as_retriever(search_kwargs={'k':6}),
                                             verbose=False, memory=memory)


print('---------------------------------------------------------------------------------')
print('Welcome to the MedBot. You are now ready to start interacting with a medical expert AI')
print('---------------------------------------------------------------------------------')

while True:
  query=input(f"Prompt:")
  if query == "exit" or query == "quit" or query == "q" or query == "f":
    print('Exiting')
    sys.exit()
  if query == '':
    continue
  result = pdf_qa({"question": query})
  print(f"Answer: " + result["answer"])
