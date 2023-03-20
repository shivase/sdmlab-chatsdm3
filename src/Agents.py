import os
from dotenv import load_dotenv
from langchain import GoogleSearchAPIWrapper
from langchain.agents import Tool ,initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from Prompt import Prompt

load_dotenv()

class Agents:

    def __init__(self):
        self.agents = {}
        self.model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                        model_name='gpt-3.5-turbo',
                        request_timeout=6,
                        max_tokens=4000,
                        temperature=0.9)

        search = GoogleSearchAPIWrapper()
        self.tools = [Tool(name = "Current Search",func=search.run,description="質問に答える必要がある場合に役立ちます")]
        self.prompt = Prompt()

    def set_prompt(self, prompt):
        self.prompt.set(prompt)

    def get(self, idx: int) -> AgentExecutor:
        if idx not in self.agents:
            self.agents[idx] = initialize_agent(
                self.tools,
                self.model,
                verbose=True,
                agent="chat-conversational-react-description",
                memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
                input_variables=['input','chat_history','llm_output'],
                prompt=self.prompt.get())

        return self.agents[idx]

    def delete(self, idx: int):
        if idx in self.agents:
            del self.agents[idx]