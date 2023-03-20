from langchain.memory import ConversationBufferMemory

class Memories:
    def __init__(self):
        self.memories = {}

    def get_memory(self, idx):
        if idx not in self.memories:
            self.memories[idx] = ConversationBufferMemory(memory_key='history')
        return self.memories[idx]

    def delete_memory(self, idx):
        if idx in self.memories:
            del self.memories[idx]