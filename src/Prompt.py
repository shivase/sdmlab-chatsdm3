from langchain.prompts import PromptTemplate

class Prompt:
    def __init__(self):
        self.prefix = ('The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n'
                            '\n'
                            'Current conversation:\n'
                            '{chat_history}\n'
                            'Human: {input}\n'
                            'AI:{llm_output}')

    def get(self):
        system_template = self.initial_file_sync_read()
        template = f"{system_template}\n{self.prefix}"
        return PromptTemplate(template=template, input_variables=['input', 'chat_history', 'llm_output'])

    def set(self, text):
        with open('initial.txt', 'w', encoding='UTF-8') as f:
            f.write(self.remove_heads(text, 1))

    def initial_file_sync_read(self):
        with open('initial.txt', 'r', encoding="UTF-8") as f:
            return f.read()

    def remove_heads(self, string, num):
        return '\n'.join(string.split('\n')[num:])