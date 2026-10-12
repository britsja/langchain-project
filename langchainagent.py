import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilySearch()

@tool
def search(query: str) -> str:
  """
  Tool that searches the internet
  Args:
    query: The query for search for
  Returns:
    The search result
  """
  print(f"Searching for {query}")
  return tavily.invoke({"query": query})

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
  )
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
  print("Langchain app started")
  result = agent.invoke({"messages": HumanMessage(content="Search for 3 open job postings in the US for a remote AI Engineer that uses Langchain")})
  print(result)

if __name__ == "__main__":
  main()

