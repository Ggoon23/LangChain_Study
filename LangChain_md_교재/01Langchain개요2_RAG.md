# RAG ?

**RAG** (Retrieval-Augmented Generation) 기법은 기존의 대규모 언어 모델(LLM)을 확장하여, 주어진 컨텍스트나 질문에 대해 더욱 정확하고 풍부한 정보를 제공하는 방법입니다. 
 - 모델의 학습 데이터에 포함되지 않은 외부 데이터를 실시간으로 검색(retrieval)하고, 
 - 이를 바탕으로 답변을 생성(generation)하는 과정을 포함합니다. 
 - 특히 환각(생성된 내용이 사실이 아닌 것으로 오인되는 현상)을 방지하고, 
 - 모델이 최신 정보를 반영하거나 더 넓은 지식을 활용할 수 있게 합니다.

왜 RAG 를 사용하나?

- 풍부한 정보 제공
  -  RAG 모델은 검색을 통해 얻은 외부 데이터를 활용하여, 보다 구체적이고 풍부한 정보를 제공할 수 있습니다.
- 실시간 정보 반영
  -  최신 데이터를 검색하여 반영함으로써, 모델이 실시간으로 변화하는 정보에 대응할 수 있습니다.
- 환각 방지
  -  검색을 통해 실제 데이터에 기반한 답변을 생성함으로써, 환각 현상이 발생할 위험을 줄이고 정확도를 높일 수 있습니다.

RAG 모델의 기본 구조

<img src='https://www.deepchecks.com/wp-content/uploads/2024/10/img-rag-architecture-model.jpg' width=800> [그림. https://www.deepchecks.com/glossary/rag-architecture/]


1. 검색 단계(Retrieval Phase)
    - 사용자의 질문이나 컨텍스트를 입력으로 받아서, 이와 관련된 외부 데이터를 검색하는 단계
    - 다양한 소스(검색 엔진, 데이터베이스 등등)에서 필요한 정보를 찾아냅니다. 
    - 검색된 데이터는 질문에 대한 답변을 생성하는데 적합하고 상세한 정보를 포함하는 것을 목표로 합니다.

2. 생성 단계(Generation Phase)
    - 검색된 데이터를 기반으로 LLM 모델이 사용자의 질문에 답변을 생성하는 단계입니다. 
    - 이 단계에서 모델은 검색된 정보와 기존의 지식을 결합하여, 주어진 질문에 대한 답변을 생성합니다.


기술적 관점

1. 임베딩 모델 설정:

    - HuggingFaceEmbeddings를 사용하여 텍스트를 벡터로 변환합니다.

2. Dataloader
    - 데이터를 준비한다.

3. 벡터 스토어 로드:
    - Chroma 벡터 스토어를 로드하고 retriever를 생성합니다.

4. 프롬프트 템플릿:

    - 검색된 문맥과 사용자 질문을 LLM에 전달할 프롬프트 템플릿을 정의합니다.

5. RAG 체인 구성:

    - retriever를 통해 문맥을 검색하고, 프롬프트 템플릿에 문맥과 질문을 넣어 LLM을 호출합니다.
    - LLM의 응답을 문자열로 변환합니다.

6. 질의응답:

    - rag_chain.invoke()를 사용하여 질문을 전달하고 답변을 출력합니다.


<img src='https://www.deepchecks.com/wp-content/uploads/2024/10/img-advanced-rag.jpg' width=800>

기술적으로 RAG 를 구현하기 위해서 LangChain Framework 를 활용한다.

# Langchain

유저의 의도에 맞는 정확한 피드백을 제공하기 위해서는 LLM, RAG, Function call을 포함한 다양한 기술을 통해 답변을 처리할 수 있는 Agent 방식을 기반으로 동작하도록 챗봇을 구현해야하는데, Agent 개발을 가장 편하게 구현할 수 있게 도와주는 대표적인 프레임워크로  Langchain을 많이 사용하고 있다.

 - Agent를 구성하는데 필요한 여러 기능 및 프로세스를 간편한 인터페이스로 연결하는 다양한 기능들을 제공
 -prompt, parser, model 등을 복합적으로 동작할 수 있다.

LangChain 패키지
 - 'langchain-core' : 주요 추상화, 인터페이스, 그리고 핵심 기능이 포함되어 있습니다.
 - 'langchain-communiy' 
- langchain-openai


LLM 과 외부도구를 사슬처럼 엮어 준다. ex) LLM 과 웹 연결
 - LLM
 - Streamlit

<img src='https://blog.streamlit.io/content/images/2023/05/schematic-1.jpeg' width=700>

### 주요 모듈

LangChain Community 는 기본 API interface 를 기반으로 제3자 모듈을 포함하고 있다.

<img src='https://pypi-camo.freetls.fastly.net/957881d9be23360c357c8aa75789ceb82553d6db/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6c616e67636861696e2d61692f6c616e67636861696e2f6d61737465722f646f63732f7374617469632f7376672f6c616e67636861696e5f737461636b5f3131323032342e737667'>

1. 모델 IO: 프롬프트, 언어모델, 출력파서
2. 데이터연결: 문서 가져오기, 변환, 임베딩, 벡터 저장소, 검색기
3. 체인: 
4. 메모리
5. 에이전트/툴

<img src='https://upstash.com/blog/langchain-explained/modules.png' width=700><br><https://upstash.com/blog/langchain-explained>

### 1. 모델 I/O

언어 모델과 상호작용하기 위한 모듈, LLM 과 상호작용은

 - LLM에 전달할 프롬프트 생성
 - 프롬프트는 입력 데이터와 검색 결과에 대한 표현
 - 답변을 받기 위해 모델 API 호출
 - 답변에 대한 출력
 - 언어모델
   - LLM, 채팅 모델, 임베딩 모델 등에 대한 API 호출
 
<img src='https://upstash.com/blog/langchain-explained/modelio.png' width=500>

### 2. 데이터연결

일반적 데이터분석 환경의 ETL(extract, transform, load) 에 해당한다.
 - 데이터 추출
 - 데이터 변환
 - 데이터 적재


ETL을 위한 구성요소
 - 문서 가져오기: 여러 출처에서 Extract
 - 문서 변환: 입력  데이터를 chunk로 분할하거나 다시 결합, 필터링 기능
 - 문서 임베딩: 텍스트를 벡터 형태로 변환
 - 벡터 저장소:
 - 검색기 retriever

<img src='https://python.langchain.com/v0.1/assets/images/data_connection-95ff2033a8faa5f3ba41376c0f6dd32a.jpg'><https://python.langchain.com/v0.1/docs/modules/data_connection/>



# --- 참고 ---



1. RAG (Retrieval-Augmented Generation) 기법
    - https://wikidocs.net/231364
2. RAG 개요
    - https://wikidocs.net/231393
3. Langchain explained
    - https://upstash.com/blog/langchain-explained
5. Langchain How-to guides
    - https://python.langchain.com/docs/how_to/
6. Text Splitter
    - https://python.langchain.com/docs/concepts/text_splitters/
    - https://python.langchain.com/docs/how_to/#text-splitters

7. https://python.langchain.com/docs/integrations/llms/llamacpp/

8. llama-cpp-python
    - https://github.com/abetlen/llama-cpp-python
