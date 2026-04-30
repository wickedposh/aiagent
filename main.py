from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Literal
from langgraph.checkpoint.mongodb import MongoDBSaver
from anthropic import Anthropic
from google import genai

load_dotenv()
g=genai.Client()
class State(TypedDict):
    messages:Annotated[list,add_messages]
llm=init_chat_model(model="gpt-4.1-mini",model_provider="openai")
def chatbot(state:State):
    response=llm.invoke(state.get("messages"))
    return {"messages": [response]}
def sample(state:State):
    print("sample node")
    return {"messages": ["hi this is a message from the sample node." ]}
graph_builder=StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("sample",sample)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot","sample")
graph_builder.add_edge("sample",END)
graph=graph_builder.compile()

update_state=graph.invoke(State({"messages":["hi my name is yuil."]}))
print("updated_state",update_state)


client=OpenAI()
class State(TypedDict):
    user_query:str
    llm_output:Optional[str]
    is_good:Optional[bool]
def chatbot2(state:State):
    response=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[ {'role' : "user","content" : state.get("user_query")} ]
    )
    state["llm_output"]=response.choices[0].message.content
    return state
def evaluation(state:State)->Literal["gemini_chat","endnode"]:
    if state.get("is_good"):
        return "gemini_chat"

    return "endnode"

def gemini_chat(state:State):
    response=client.chat.completions.create(
        model="gemma-3",
        messages=[{'role':"user","content":state.get("user_query")}]
    )
    state["llm_output"]=response.choices[0].message.content
    return state
def endnode(state:State):
    return state
graph_builder2=StateGraph(State)
graph_builder2.add_node("chatbot2",chatbot2)
graph_builder2.add_node("gemini_chat",gemini_chat)
graph_builder2.add_node("endnode",endnode)
graph_builder2.add_edge(START,"chatbot2")
graph_builder2.add_conditional_edges("chatbot2",evaluation)
graph_builder2.add_edge("chatbot2","endnode")
graph_builder2.add_edge("endnode",END)

def compile_with_checkpoint(checkpointer):

    return graph_builder2.compile(checkpointer=checkpointer)
DB_URL = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URL) as checkpointer:
    graph_check=compile_with_checkpoint(checkpointer=checkpointer)
    config={
        "configurable":{"thread_id":"Math"}
    }
    for chunk in graph_check.stream(
        State({"user_query":"what is 2+2?"}),
        config,
        stream_mode="updates"):
        for node, state_update in chunk.items():
            if "llm_output" in state_update:
                print(f"Node {node} output: {state_update['llm_output']}")



## I want to apply voting in machine learning algorithm among AI providers. But
##currenlty only just compare two and extract the information in common.
ann=Anthropic()

class State_C(TypedDict):
    user_query : str
    openaioutput:Optional[str]
    geminioutput:Optional[str]
    comparison_result:Optional[str]
def chatbot3(state:State):
    response=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[ {'role' : "user","content" : state.get("user_query")} ]
    )
    state["openaioutput"]=response.choices[0].message.content
    return state
def Claude(state:State):
    response=ann.messages.create(
        model="claude-3-opus-20240229",
        messages=[{'role':"user","content":state.get("user_query")}],
        max_tokens=10
    )
    state["geminioutput"]=response.content[0].text
    return state
def gemini_chatC(state:State):
    response=g.models.generate_content(
        model="gemini-2.0-flash",
              contents="user_query")
    state["geminioutput"]=response.text
    return state
def endnode(state:State):
    return state
def comparison(state:State):
    o1=state["openaioutput"]
    o2=state["geminioutput"]
    PROMPT=f"""
        There are two texts answering to the same question:
        Text1:{o1}
        Text2:{o2}
        Please identify and extract the information that appears in BOTH responses.
        Focus on facts, conclusions or key points that are consistent across both.
        
    """

    comparison_result=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{'role':"user","content":PROMPT}],
    )
    state["comparison_result"]=comparison_result.choices[0].message.content
    return state



graph_builder3=StateGraph(State)
graph_builder3.add_node("chatbot3",chatbot3)
graph_builder3.add_node("gemini_chatC",gemini_chatC)
graph_builder3.add_node("endnode",endnode)
graph_builder3.add_node("comparison",comparison)
graph_builder3.add_edge(START,"chatbot3")
graph_builder3.add_edge(START,"gemini_chatC")
graph_builder3.add_edge("chatbot3","comparison")
graph_builder3.add_edge("gemini_chatC","comparison")
graph_builder3.add_edge("comparison","endnode")
graph_builder3.add_edge("endnode",END)
def compile_with_checkpoint(checkpointer):

    return graph_builder3.compile(checkpointer=checkpointer)
DB_URL = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URL) as checkpointer:
    graph_check=compile_with_checkpoint(checkpointer=checkpointer)
    config={
        "configurable":{"thread_id":"Math"}
    }
    for chunk in graph_check.stream(
        State({"user_query":"what is 2+2?"}),
        config,
        stream_mode="updates"):
        for node, state_update in chunk.items():
            if "llm_output" in state_update:
                print(f"Node {node} output: {state_update['llm_output']}")
