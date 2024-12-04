from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.schema import HumanMessage, AIMessage
from collections import Counter
import logging

class ConversationChain:
    def __init__(self, memory, summarizer, ensemble_retriever, movie_name_retriever, prompt_template, conversation_pipe):
        self.memory = memory
        self.summarizer = summarizer
        self.ensemble_retriever = ensemble_retriever
        self.movie_name_retriever = movie_name_retriever
        self.prompt_template = prompt_template
        self.conversation_pipe = conversation_pipe
        self.logger = logging.getLogger(__name__)

        # Initialize the Conversation Chain Pipeline
        self.conversation_chain = self.build_conversation_chain()

    def extract_movie_name_with_majority(self, results):
        if results and len(results) > 0:
            movie_counts = Counter(
                result.metadata['movie_name'] 
                for result in results 
                if 'movie_name' in result.metadata
            )
            return movie_counts.most_common(1)[0][0] if movie_counts else None
        return None

    def build_conversation_chain(self):
        # Define RunnableParallel for initial inputs
        runnable_parallel = RunnableParallel({
            "instruction": RunnableLambda(lambda inputs: inputs["instruction"]),
            "context": RunnableLambda(self.retrieve_context),  # Retrieve and format documents as context
            "question": RunnableLambda(lambda inputs: inputs["question"]),   # Pass the user's question through
            "chat_history": RunnableLambda(self.format_chat_history)
        })

        # Define the pipeline chaining
        conversation_chain = (
            runnable_parallel
            | self.prompt_template  # Use a prompt to handle the user question
            | self.conversation_pipe  # Model to generate the answer
            | StrOutputParser()  # Parse the model's output as a string
        )

        self.logger.info("Built conversation chain pipeline")
        return conversation_chain

    def format_chat_history(self, inputs):
        chat_history = self.memory.load_memory_variables({}).get('chat_history', [])
        chat_content = " ".join(
            msg.content for msg in chat_history if isinstance(msg, (HumanMessage, AIMessage))
        )
        if chat_content:
            summarized_chat = self.summarizer(chat_content)[0]['summary_text']
            self.logger.debug(f"Summarized chat history: {summarized_chat}")
            return summarized_chat
        return ""

    def retrieve_movie_name(self, question):
        # Use movie_name_retriever to find the most relevant movie name in the question
        results = self.movie_name_retriever.get_relevant_documents(question)
        movie_name = self.extract_movie_name_with_majority(results)
        if movie_name:
            self.logger.info(f"Extracted movie name: {movie_name}")
        else:
            self.logger.warning("No movie name extracted from the question.")
        return movie_name

    def retrieve_context(self, inputs):
        question = inputs.get("question", "")
        # Extract movie name using movie_name_retriever
        movie_name = self.retrieve_movie_name(question)

        # Configure ensemble_retriever based on movie_name
        if movie_name:
            # Apply filter for movie_name in ensemble retriever
            self.ensemble_retriever.search_kwargs["filter"] = {"movie_name": {"$eq": movie_name}}
            self.logger.info(f"Configured ensemble retriever with movie filter: {movie_name}")
        else:
            # Remove any existing filter
            self.ensemble_retriever.search_kwargs.pop("filter", None)
            self.logger.info("No movie filter applied to ensemble retriever.")

        # Retrieve documents using ensemble retriever
        documents = self.ensemble_retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in documents])
        self.logger.info(f"Retrieved {len(documents)} documents for context.")
        return context

    def handle_conversation(self, inputs):
        response = self.conversation_chain.run(inputs)
        question = inputs['question']
        self.update_memory(question, response)
        return response

    def update_memory(self, question, response):
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)
        self.logger.info("Updated conversation memory.")
