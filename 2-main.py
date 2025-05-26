from langchain_community.embeddings import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal, Annotated
#from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
import yaml

#with open("config.yaml", "r") as file:
#    config = yaml.safe_load(file)


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OpenAI_API_Key')

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
model = ChatOpenAI()

vector_store = Chroma(collection_name='knowledge_graph_collection', persist_directory="./ChromaNew", embedding_function=embeddings)

llm = init_chat_model("gpt-4o", 
                      model_provider="openai", 
                      temperature=0.0, 
                      max_tokens=1000
                      )

class MessageClassifier(BaseModel):
    message_type: Literal["DistributedCompute", "QueryCategorization", "SatelliteImagery", "search_all"] = Field(
        ...,
        description="Classify if the message requires content from Distributed Compute in Deep Learning, Query Categorization, Satellite Imagery or None making retriver search all the chunks."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    if not state.get("messages"):
        return state

    last_message = state["messages"][-1]

    # Ensure message has proper structure and content
    if not isinstance(last_message, dict) or "content" not in last_message or not last_message.content:
        return state  # Skip classification

    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'DistributedCompute': if it asks for distributed computing, deep learning training with distributed computing, or distributed computing platforms
            - 'QueryCategorization': if it asks for categorizing queries, query optimization, or query performance
            - 'SatelliteImagery': if it asks for satellite imagery analysis, remote sensing, or geospatial data
            - 'search_all': if it does not fit any of the above categories
            """
        },
        {"role": "user", "content": last_message.content}
    ])

    return {**state, "message_type": result.message_type}



def router(state: State):
    message_type = state.get("message_type", "search_all")
    if message_type == "DistributedCompute":
        # Route to DistributedCompute Retriver agent
        return {"next": "RetrieverAgent1"}
    elif message_type == "QueryCategorization":
        # Route to QueryCategorization Retriver agent
        return {"next": "RetrieverAgent2"}
    elif message_type == "SatelliteImagery":
        # Route to SatelliteImagery Retriver agent
        return {"next": "RetrieverAgent3"}

    return {"next": "RetrieverAgentAll"}  # Default to search_all if no match


def RetrieverAgent1(state: State):

    last_message = state["messages"][-1]
    user_input = last_message.content

    # Create department-specific retriever
    retriever = vector_store.as_retriever(search_kwargs={"filter": {"department": "DistributedCompute"}})

    # Prompt template for context-based QA
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:" \
        {context}
        
        Question: {question}"""
    )

    # Build retrieval chain
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    # Invoke retrieval chain
    reply = retrieval_chain.invoke(user_input)

    # Return updated state with appended assistant message
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": reply}]
    }

def RetrieverAgent2(state: State):

    last_message = state["messages"][-1]
    user_input = last_message.content

    retriever = vector_store.as_retriever(search_kwargs={"filter": {"department": "SatelliteImagery"}})

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:" \
        {context}
        
        Question: {question}"""
    )

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    reply = retrieval_chain.invoke(user_input)

    # Return updated state with appended assistant message
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": reply}]
    }


def RetrieverAgent3(state: State):

    last_message = state["messages"][-1]
    user_input = last_message.content

    retriever = vector_store.as_retriever(search_kwargs={"filter": {"department": "QueryCategorization"}})

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:" \
        {context}
        
        Question: {question}"""
    )

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    reply = retrieval_chain.invoke(user_input)

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": reply}]
    }

def RetrieverAgentAll(state: State):

    last_message = state["messages"][-1]
    user_input = last_message.content

    # Create department non-specific retriever
    retriever = vector_store.as_retriever()

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:" \
        {context}
        
        Question: {question}"""
    )

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    reply = retrieval_chain.invoke(user_input)

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": reply}]
    }


def create_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", classify_message)
    graph_builder.add_node("router", router)
    graph_builder.add_node("RetrieverAgent1", RetrieverAgent1)
    graph_builder.add_node("RetrieverAgent2", RetrieverAgent2)
    graph_builder.add_node("RetrieverAgent3", RetrieverAgent3)
    graph_builder.add_node("RetrieverAgentAll", RetrieverAgentAll)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"RetrieverAgent1": "RetrieverAgent1", "RetrieverAgent2": "RetrieverAgent2", "RetrieverAgent3": "RetrieverAgent3", "RetrieverAgentAll": "RetrieverAgentAll"}
    )

    graph_builder.add_edge("RetrieverAgent1", END)
    graph_builder.add_edge("RetrieverAgent2", END)
    graph_builder.add_edge("RetrieverAgent3", END)
    graph_builder.add_edge("RetrieverAgent4", END)

    graph = graph_builder.compile()
    return graph
graph = create_graph()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()

