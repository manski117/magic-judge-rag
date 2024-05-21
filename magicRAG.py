### import statements ###
import os #core python module to allow env files to be loaded up
# pip install python-dotenv
from dotenv import load_dotenv, dotenv_values
load_dotenv()
import json
import requests
from pprint import pprint

###### System Prompt and Main User Query ########
# og_user_query = "Can you tap a land to generate mana in response to your opponent declaring a spell on the stack?"
# og_user_query = "Typhoid Rats has deathtouch and a power and toughness of 1/1. If it blocks Ainok Tracker which has first strike and a power and toughness of 3/3, does the Aniok Tracker die, or does it survive because it has first strike?"
# og_user_query = "If I am at 2 life remaining and I take damage from an unblocked Aether Chaser, do I lose the game?"
# og_user_query = "My opponent controls both Sanguine Bond and Exquisite Blood and attacks me with Typhoid Rats. Can you please compare the card interactions for the cards Sanguine Bond and Exquisite Blood and tell me if taking one damage from an unblocked Typhoid Rats would cause me to lose an infinite amount of life?"
### llm initial system message ###
llm_system_prompt = "You are a powerful AI assistant, but you don't know the rules of the magic the gathering card game. You will be given tools to help supplement your knowledge of card text, the rules of the game, and judge rulings on card interactions. Your job will be to be a specialized AI assistant to answer questions as a judge for the magic the gathering card game to help palyers understand the rules of the game and complex interactions that may arise."


class MagicRAG:
    def __init__(self, og_user_query, llm_system_prompt=llm_system_prompt):
        self.key = os.getenv("MY_OPENAI_API_KEY")
        self.og_user_query = og_user_query
        self.llm_system_prompt = llm_system_prompt
    

    def output(self):
        ###### setup model ######
        key = os.getenv("MY_OPENAI_API_KEY")
        # install and initialize the llm
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo-preview", temperature=0)

        # create initial prompt with scratchpad for agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    llm_system_prompt,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        ###### setup tools to get data ######
        from langchain.agents import tool

        # tool for looking up card data based on name
        @tool
        def get_card_data(card_name: str):
            """Returns all data for a magic the gathering card that you provide the name of.
            If you need information about an individual card's details or specific rulings, you must use this tool!
            """
            try:
                # Format the card name for URL: replace spaces with plus signs
                formatted_card_name = card_name.replace(' ', '+')
                url = f"https://api.scryfall.com/cards/named?fuzzy={formatted_card_name}"
                
                # Make the request to the Scryfall API
                response = requests.get(url)
                
                # Check if the response was successful
                if response.status_code == 200:
                    cardData = response.json()
                    return cardData
                else:
                    # If the response was not successful, raise an exception with the status code
                    response.raise_for_status()

            except requests.exceptions.HTTPError as http_err:
                # Handle HTTP errors that could occur if the card is not found or API issues
                raise ValueError(f"HTTP error occurred: {http_err}")  # Or use a more specific message
            except requests.exceptions.RequestException as err:
                # Handle other possible exceptions such as a network problem (e.g., DNS failure, refused connection, etc)
                raise RuntimeError(f"Request error occurred: {err}")
            except Exception as e:
                # Handle unforeseen errors
                raise RuntimeError(f"An unexpected error occurred: {str(e)}")



        ### embed data for retriever tool
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader("./data/compRules.txt")
        rulesDoc = loader.load() 

        # import embedding model
        # pip install faiss-cpu
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=key)

        # build our index now that the embedding model has been imported and the vector store installed.
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter()
        splitRulesDocs = text_splitter.split_documents(rulesDoc)
        vector = FAISS.from_documents(splitRulesDocs, embeddings)

        # create a retrieval chain now that the data has been indexed in a vector store
        # this chain takes an incoming question, looks up relevant docs, then passes the docs along with the original question to the LLM and ask it to finally answer the original question
        from langchain.chains.combine_documents import create_stuff_documents_chain
        retrievalPrompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, retrievalPrompt)
        #but that is just the chain. Now we need to create the retriever itself, 
        #which will actually dynamically select the most releavant docs and pass those in. 

        #create actual retriever
        from langchain.chains import create_retrieval_chain
        retriever = vector.as_retriever()
        #make chain that combines reriever and document_chain chaining question, contetxt, prompt, and LLM
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # wrap this all into a tool for an agent to use
        @tool
        def retrieve_game_rules(userQuery: str):
            """Search for information about magic the gathering game rules. For any questions about the rules of the magic the gathering card game, you must use this tool!"""
            return retrieval_chain.invoke({"input": userQuery})
        # pprint(retrieve_game_rules.invoke(og_user_query)) #uncomment to test tool


        ### setup our agent to empower the LLM to decide ###
        # necessary import statements
        from langchain.agents.format_scratchpad.openai_tools import(
            format_to_openai_tool_messages,
        )
        from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
        from langchain.agents import AgentExecutor

        # put our tools in a toolbox array for the agent to use
        tools = [retrieve_game_rules, get_card_data]

        # Bind tools to LLM so that it knows which tools it can use
        llm_with_tools = llm.bind_tools(tools)

        #create the actual agent
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # consume the agents stream of actions and print just the final result:
        agent_stream = agent_executor.stream({"input": self.og_user_query})

        # Consume the generator to get the last item, which should be the final answer
        final_answer = None
        for step in agent_stream:
            final_answer = step  # This will end up being the last item in the stream

        # Now you can print or use the final answer
        print('hello from agentStream method of MagicRAG class from magicRAG.py! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        # print(final_answer['output'])
        llm_final_output = final_answer['output']
        return llm_final_output

