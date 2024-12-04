from langchain.memory import ConversationBufferMemory
import logging

class MemoryManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def initialize_memory(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.logger.info("Initialized conversation memory")
        return memory
