import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def main():
  print("Starting langchain program")
  information = """
    Manchester City has been found guilty by an independent commission of breaking Premier League financial rules over a nine-season period and has officially lodged an appeal against the verdict.
    The Verdict
    An independent disciplinary commission found Manchester City guilty of nearly all of the 115 charges brought against them relating to the 2009–10 through 2017–18 seasons.
    The findings revealed that the club used "sham" contracts to disguise over £830 million in secret funding from their Abu Dhabi ownership, artificially inflating revenues and hiding the true state of their finances.
    The commission also upheld multiple charges for the club's failure to cooperate with the investigation.
    The Club's Response
    Manchester City strongly denies all accusations, calling the commission's conclusions "unsafe" and claiming they contain material errors of law and fact.
    The club formally lodged a comprehensive appeal, maintaining they possess a body of "irrefutable evidence" to clear their name.
    Potential Consequences
    Sanctions have not yet been finalized.
    Rival clubs and pundits are debating the appropriate punishment, with suggestions ranging from massive points deductions and stripping of past titles to relegation or potential expulsion from the Premier League if the appeal fails.
    the appeal process is expected to continue progressing over the coming months.
  """

  summary_template =  """
    given the information {information} about a club in the English football premier league, I want you to create:
    1. A short summary.
    2. Two interesting facts about them.
  """

  summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
  )

  llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
  )

  chain = summary_prompt_template | llm

  response = chain.invoke(input={"information": information})
  print(response.content)

if __name__ == "__main__":
  main()

