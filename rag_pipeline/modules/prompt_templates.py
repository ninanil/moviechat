from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
import logging

class PromptTemplateManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_chat_prompt_template(self):
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            """You are an experienced assistant specializing in question-answering tasks.
Use the context to answer the following question.
If you're unsure, say 'I don't know.'
Instruction: {instruction}
"""
        )

        human_message_prompt_template = HumanMessagePromptTemplate.from_template(
            "Question: {question}"
        )

        context_message_prompt_template = AIMessagePromptTemplate.from_template(
            "Chat history: {chat_history}\nContext: {context}"
        )

        chat_prompt_template = ChatPromptTemplate(
            input_variables=['instruction', 'question', 'chat_history', 'context'],
            messages=[
                system_message_prompt_template,
                context_message_prompt_template,
                human_message_prompt_template
            ]
        )
        self.logger.info("Created chat prompt template")
        return chat_prompt_template
