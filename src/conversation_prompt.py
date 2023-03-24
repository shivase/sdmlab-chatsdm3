
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import ConversationalAgent

load_dotenv()
SYSTEM_PREFIX_FILE=os.environ.get("SYSTEM_PREFIX_FILE")

SUMMARY_TEMPLATE = """
会話内容を順次要約し、前回の要約に追加して新たな要約を返してください。

### 現在の要約

{summary}

### 新しい会話

{new_lines}

### 新しい要約

"""

CONVERSATION_PREFIX = "人間と会話をおこない、できるだけ良い答えを出すように以下の質問に答えてください。次のツールにアクセスできます"
CONVERSATION_SUFFIX = """初めて下さい"

{chat_history}
Human: {input}
AI: {agent_scratchpad}
"""

QA_TEMPLATE = """Use the following pieces of context to answer the question at the end.
{context}
Question: {question}
Helpful Answer:"""

class ConversationPrompt:

    def __init__(self,tools):
        self.tools = tools

    def conversation(self):
        system_file = self.get_system_prefix()
        prefix = f"{system_file}\n{CONVERSATION_PREFIX}"

        return ConversationalAgent.create_prompt(
            self.tools,
            prefix=prefix,
            suffix=CONVERSATION_SUFFIX,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )

    def summary(self):
        return PromptTemplate(
            template=SUMMARY_TEMPLATE,
            input_variables=['summary', 'new_lines',]
        )

    def q_a(self):
        return PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])

    def set_system_prefix(self, text):
        with open(SYSTEM_PREFIX_FILE, 'w', encoding='UTF-8') as system_prefix_file:
            system_prefix_file.write(self.remove_heads(text, 1))

    def get_system_prefix(self):
        if os.path.exists(SYSTEM_PREFIX_FILE) :
            with open(SYSTEM_PREFIX_FILE, 'r', encoding="UTF-8") as system_prefix_file:
                return system_prefix_file.read()
        else:
            return ''

    def remove_heads(self, string, num):
        return '\n'.join(string.split('\n')[num:])