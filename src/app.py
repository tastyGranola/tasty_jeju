from chain import *
import streamlit as st
import pandas as pd
from langchain.schema import(
    AIMessage,
    HumanMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from typing import List
import os
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    raise ValueError("Missing Google API key!")


model_name = "jhgan/ko-sroberta-multitask"

# Chroma DB를 생성하기 위한 embedding
class MyEmbeddings:
        def __init__(self, model_name):
            self.model = SentenceTransformer(model_name, trust_remote_code=True)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self.model.encode(t).tolist() for t in texts]

        def embed_query(self, query: str) -> List[float]:
            return self.model.encode(query).tolist()

embeddings = MyEmbeddings(model_name)

# 모델 불러오기
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Chroma DB 불러오기
persist_directory = "./chroma_store_modified"
chroma_store = Chroma(embedding_function=embeddings,persist_directory=persist_directory)

# 전처리된 신한카드 데이터 불러오기
basic_data = pd.read_csv('./total_basic_data_modified.csv',index_col=0)

# 검색형 질문에 응답하기 위한 agent 생성하기
agent = create_pandas_dataframe_agent(llm,df=basic_data,verbose=True,
                         allow_dangerous_code=True,prefix="차근차근 생각해보자. '장소'와 '업종'은 반드시 맞아야해.")

def main():
  st.set_page_config(page_title="오늘은 어떤걸 드시고 싶으신가요?")
  global select_city, select_dong, select_category
  st.header("당신이 찾는 식당은?")
  st.sidebar.title("당신이 찾는 식당은?")
  select_city = []

  # 도시 선택
  select_city.append(st.sidebar.selectbox(
      '시를 선택하세요',
      ['전지역'] + list(basic_data['시'].unique())
  ))

  # 도시를 선택했다면 동 선택
  if '전지역' not in select_city:
    filtered_data = basic_data[basic_data['시'] == select_city[0]]
    select_dong = st.sidebar.multiselect(
        '동을 선택하세요. 복수선택 가능',
        options=list(filtered_data['동'].unique())
    )
  
  # 업종 선택
  select_category = st.sidebar.multiselect(
      '업종을 선택하세요. 복수선택 가능',
      options=sorted(basic_data['업종'].unique())
  )

  # 사용자 입력시 실행
  if query := st.chat_input():
    if 'messages' not in st.session_state:
     st.session_state['messages'] = []
    st.session_state.messages.append(HumanMessage(query))
    with st.chat_message("user"):
      st.markdown(query)

    # 지역을 선택하지 않았을 경우 전지역 선택
    try:
      if '전지역' in select_city:
        select_city = list(basic_data['시'].unique())
    except:
      select_city = list(basic_data['시'].unique())

    # 동을 선택하지 않았을 경우 모든 동 선택
    try:
      if not select_dong:
        select_dong = list(basic_data['동'].unique())
    except:
      select_dong = list(basic_data['동'].unique())

    # 업종을 선택하지 않았을 경우 모든 업종 선택
    try:
      if not select_category:
        select_category = list(basic_data['업종'].unique())
    except:
      select_category = list(basic_data['업종'].unique())
    
    # 응답도출
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        response = full_chain.invoke({"요구사항" : query})  # 아래의 route와 full_chain 참조. full_chain 객체가 route함수를 사용  
        placeholder = st.container()
        placeholder.markdown(response)
    st.session_state.messages.append(AIMessage(response))

def route(query: dict) -> RunnableSequence:
  '''
  이 함수는 쿼리의 'query_type'을 분석하여, "메뉴" 또는 "상황"과 관련된 경우 
  음식 관련 체인을 실행해 RAG를 통해 관련된 식당을 가져오고, 
  "정보"와 관련된 쿼리일 경우, dataframe agent를 만들어서 대회 평가 항목 1번에
  대한 응답을 하는 체인이 실행됩니다.

  나머지 이상한 쿼리에 대해서는 null_chain을 실행합니다. null chain은 사용자에게
  다시 제대로 된 쿼리를 유도합니다.

  Return:
      RunnableSequence: 쿼리의 타입에 맞는 체인을 반환하며, 
      이는 full_chain의 인자로 전달됩니다. 
  '''
  global basic_data

  # 사용자의 질문이 메뉴 혹은 상황을 요구한다면 RAG를 통한 base chain기반 응답
  if '메뉴'in query['query_type'].lower() or '상황' in query['query_type'].lower():
    food = q_to_food_chain.invoke(query['요구사항']).strip()
    filters = {
    "$and": [
        {'동': {"$in": select_dong}},
        {'시': {"$in": select_city}},
        {'업종' : {"$in": select_category}}
    ]
    }
    retriever = chroma_store.as_retriever(search_type="similarity", search_kwargs={
        "k": 20,
        "filter": filters
    })
    
    # RAG로 연관 있는 식당이름들 추출
    results = retriever.get_relevant_documents(f'{food} 먹고싶어')
    relevant_stores = get_store_names_from_result(results)

    # 신한 카드 데이터 중 RAG를 통해 뽑힌 식당 이름들만 추출 
    filtered_data = basic_data.loc[relevant_stores,:]

    # 신한카드 데이터에서 정보를 추출해 하나의 텍스트 덩어리로 변환 (gemini에게 input으로 넣어주기 위함)
    reference_info = ""
    for idx, row in filtered_data.iterrows():
      row_text = ""
      for key, value in zip(row.index, row):
        row_text += f'{key}:{value}\n'
      reference_info += f"{row_text}\n\n"

    base_chain_with_reference_info = (
    RunnableLambda(lambda inputs : {"식당정보": reference_info, "요구사항" : query['요구사항']}) | base_chain
    )
    return base_chain_with_reference_info
  
  # 사용자의 query가 검색형 질문이었다면 dataframe agent를 이용한 chain 실행
  elif '정보' in query['query_type'].lower():
    result = agent.invoke(query['요구사항'])['output']

    process_agent_output_chain_with_output = (
        RunnableLambda(lambda inputs : {"요구사항": result}) | process_agent_output_chain
    )
    return process_agent_output_chain_with_output

  # 사용자의 query가 비정상적이라면
  else:
    return null_chain

full_chain = (
    {"query_type" : query_classify_chain, "요구사항" : itemgetter("요구사항")} |
    RunnableLambda(
        route
    )
    | StrOutputParser()
) 

def get_store_names_from_result(docs : List) -> List[str]:
  """
  RAG할 때 vector DB에서 쿼리와 연관 있는 document들을 가져오는데 저희는 어차피
  여기서 식당 이름 정보만 필요해서 식당 이름만 뽑는 함수입니다.

  Return:
      List[str]
  """
  results = [doc.metadata['full_store_name'] for doc in docs]
  return results

@st.cache_data
def load_data(path):
  st.header("Ask your CSV")
  df = pd.read_csv(path)
  return df

if __name__ == "__main__":
  main()
