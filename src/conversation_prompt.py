from langchain.prompts import PromptTemplate
from langchain.agents import ConversationalAgent

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

class ConversationPrompt:

    def __init__(self,tools):
        self.tools = tools

    def conversation(self):
        return ConversationalAgent.create_prompt(
            self.tools,
            prefix=CONVERSATION_PREFIX,
            suffix=CONVERSATION_SUFFIX,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )

    def summary(self):
        return PromptTemplate(
            template=SUMMARY_TEMPLATE,
            input_variables=['summary', 'new_lines',]
        )

    def set(self, text):
        with open('initial.txt', 'w', encoding='UTF-8') as f:
            f.write(self.remove_heads(text, 1))

    def initial_file_sync_read(self):
        with open('initial.txt', 'r', encoding="UTF-8") as f:
            return f.read()

    def remove_heads(self, string, num):
        return '\n'.join(string.split('\n')[num:])