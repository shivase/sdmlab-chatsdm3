import os
from dotenv import load_dotenv
from langchain import LLMMathChain, LLMChain
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.agents import Tool , AgentExecutor, ConversationalAgent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import ChatVectorDBChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from conversation_prompt import ConversationPrompt
from document_loader import DocumentLoader

load_dotenv()

class QAAgent:

    def __init__(self):
        self.agents = {}
        self.summary_llm = OpenAI(
                                temperature=0,
                                max_tokens=1000,
                                verbose=True)
        self.qa_llm = ChatOpenAI(
                            openai_api_key=os.environ['OPENAI_API_KEY'],
                            model_name='gpt-3.5-turbo',
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
        print("loading document...");
        self.vectorstore = DocumentLoader().load('document');
        print("done loading document")

    def set_prefix(self, prompt):
        self.prompt.set_system_prefix(prompt)

    def get_executor(self, idx: int) -> AgentExecutor:
        if idx not in self.agents:

            self.agents[idx] = ChatVectorDBChain.from_llm(
                llm=self.qa_llm,
                chain_type="stuff",
                vectorstore=self.vectorstore,
                qa_prompt=self.prompt.q_a(),
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                verbose=True,
            )

            #self.agents[idx] = AgentExecutor.from_agent_and_tools(
            #    agent=ConversationalAgent(
            #            llm_chain=qa,
            #            allowed_tools=self.tool_names),
            #    memory=ConversationSummaryMemory(
            #                        llm=self.summary_llm,
            #                        memory_key='chat_history',
            #                        prompt=self.prompt.summary(),
            #                        return_messages=True),
            #    tools=self.tools)

        return self.agents[idx]

    def delete(self, idx: int):
        if idx in self.agents:
            del self.agents[idx]