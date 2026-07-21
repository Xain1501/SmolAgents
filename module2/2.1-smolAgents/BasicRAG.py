import os
from dotenv import load_dotenv
from smolagents import CodeAgent, Tool, InferenceClientModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
hf_token = os.getenv("HF_TOKEN")


class EventPlanningRetrieverTool(Tool):
    name = "EventPlanningRetriever"
    description = (
        "Uses BM25 search to retrieve relevant wedding planning ideas."
    )

    inputs = {
        "query": {
            "type": "string",
            "description": "A query related to wedding planning or wedding themes.",
        }
    }

    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.retriever = BM25Retriever.from_documents(
            documents=docs,
            k=5,
        )

    def forward(self, query: str):
        assert isinstance(query, str), "Query must be a string."

        retrieved_docs = self.retriever.invoke(query)

        return "\nRetrieved Ideas:\n" + "\n".join(
            [
                f"\n===== Idea {i+1} =====\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

event_ideas = [
    {
        "text": "A classic royal wedding theme with elegant white and gold décor, crystal chandeliers, and floral centerpieces.",
        "source": "Wedding Themes",
    },
    {
        "text": "A rustic outdoor wedding featuring wooden décor, fairy lights, wildflowers, and a countryside-inspired atmosphere.",
        "source": "Venue & Decor",
    },
    {
        "text": "A beach wedding with pastel-colored decorations, seashell accents, tropical flowers, and a sunset ceremony.",
        "source": "Destination Wedding Ideas",
    },
    {
        "text": "A modern minimalist wedding using neutral tones, sleek table settings, geometric decorations, and contemporary floral arrangements.",
        "source": "Modern Wedding Trends",
    },
    {
        "text": "A traditional South Asian wedding with vibrant floral decorations, mehndi, live cultural performances, and an elaborate wedding feast.",
        "source": "Cultural Wedding Ideas",
    },
]

source_docs = [
    Document(
        page_content=doc["text"],
        metadata={"source": doc["source"]},
    )
    for doc in event_ideas
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
)

docs_processed = text_splitter.split_documents(source_docs)

event_planning_retriever = EventPlanningRetrieverTool(docs_processed)


model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token,
)

agent = CodeAgent(
    tools=[event_planning_retriever],
    model=model,
)


response = agent.run(
    "Find ideas for a luxury wedding-themed event, including entertainment, catering, and decoration options."
)

print(response)