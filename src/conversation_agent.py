import os
from dotenv import load_dotenv
from langchain import LLMMathChain, LLMChain
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.agents import Tool , AgentExecutor, ConversationalAgent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from conversation_prompt import ConversationPrompt

load_dotenv()

class ConversationAgent:

    def __init__(self):
        self.agents = {}
        self.summary_llm = OpenAI(
                                temperature=0,
                                max_tokens=1000,
                                verbose=True)
        self.reply_llm = ChatOpenAI(
                            openai_api_key=os.environ['OPENAI_API_KEY'],
                            model_name='gpt-3.5-turbo',
                            max_tokens=2500,
                            verbose=True,
                            request_timeout=5,
                            max_retries=3,
                            temperature=0.9)
        self.tools = [
            Tool(
                name = "Search",
                func=GoogleSearchAPIWrapper().run,
                description="現在起こっている出来事に関する質問に答える際に役立ちます"
                ),
            Tool(
                name = "Calculator",
                func=LLMMathChain(llm=OpenAI(
                                        temperature=0,
                                        max_tokens=500,
                                        verbose=True)).run,
                description="数学に関する質問に答える際に役立ちます"
                )
            ]
        self.tool_names = [tool.name for tool in self.tools]
        self.prompt = ConversationPrompt(self.tools)

    def set_prompt(self, prompt):
        self.prompt.set(prompt)

    def get_executor(self, idx: int) -> AgentExecutor:
        if idx not in self.agents:
            self.agents[idx] = AgentExecutor.from_agent_and_tools(
                agent=ConversationalAgent(
                        llm_chain=LLMChain(
                            llm=self.reply_llm,
                            prompt=self.prompt.conversation(),
                            verbose=True),
                        allowed_tools=self.tool_names),
                memory=ConversationSummaryMemory(
                                    llm=self.summary_llm,
                                    memory_key='chat_history',
                                    prompt=self.prompt.summary(),
                                    return_messages=True),
                tools=self.tools)

        return self.agents[idx]

    def delete(self, idx: int):
        if idx in self.agents:
            del self.agents[idx]