import warnings
warnings.filterwarnings("ignore")

import os
import re
import psycopg
import pandas as pd
from itertools import islice
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()   # reads OPENAI_API_KEY and DATABASE_URL from .env

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
FILES_FOLDER = "files"
DATABASE_URL = os.getenv("DATABASE_URL")     # postgresql+psycopg://user:password@localhost:5432/ragdb
COLLECTION_NAME = "documents"

MAX_ROWS = 2000            # only read the first 2000 rows of a csv/excel
TOP_K = 4                  # FIX 3: fewer chunks = less noise (was 6)
SCORE_LIMIT = 0.5          # FIX 3: stricter (was 0.6). Cosine distance, lower = better
MIN_CHUNK_LENGTH = 50      # FIX 3: throw away tiny junk chunks (prose files only)

SUPPORTED = (".pdf", ".txt", ".docx", ".csv")
FILE_TYPES = ["pdf", "docx", "txt", "csv"]
PROSE_TYPES = ["pdf", "docx", "txt"]

RRF_K = 60          # standard constant for Reciprocal Rank Fusion, leave as is
CANDIDATES = 10     # how many results each search returns before merging

# Words that appear in many chunks and do not help find a specific one.
# Edit this list for your own data.
IGNORE_WORDS = {"list", "show", "give", "tell", "find", "provide", "name", "names",
                "title", "titles", "genre", "genres", "belong", "belongs",
                "please", "could", "would", "should", "me",
                "return", "returns", "details", "detail", "get", "display", "fetch",
                "all", "every", "each", "information", "info", "about",
                "records", "record", "rows", "row", "entries", "data",
                "related", "matching"}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    use_jsonb=True,
)

def create_keyword_index():
    """One-time (safe to repeat): makes the word search fast."""
    plain_url = DATABASE_URL.replace("postgresql+psycopg", "postgresql")
    with psycopg.connect(plain_url) as conn:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_keyword_idx
            ON langchain_pg_embedding
            USING GIN (to_tsvector('english', document))
        """)


create_keyword_index()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


# ------------------------------------------------------------------
# INGESTION: one file -> chunks -> Postgres
# ------------------------------------------------------------------
def delete_old_chunks(filename):
    """If the same file is uploaded again, remove its old chunks first (no duplicates)."""
    plain_url = DATABASE_URL.replace("postgresql+psycopg", "postgresql")
    with psycopg.connect(plain_url) as conn:
        conn.execute(
            """
            DELETE FROM langchain_pg_embedding
            WHERE cmetadata->>'filename' = %s
              AND collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = %s)
            """,
            (filename, COLLECTION_NAME),
        )


def ingest_file(path):
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        docs = PyPDFLoader(path).load()
        for d in docs:   # page-navigation text from the web copy of the book
            d.page_content = d.page_content.replace("previous | Table of Contents | next", "")

    elif ext == ".txt":
        docs = TextLoader(path, encoding="utf-8").load()

    elif ext == ".docx":
        docs = Docx2txtLoader(path).load()

    elif ext == ".csv":
        with open(path, encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","

        loader = CSVLoader(path, encoding="utf-8", csv_args={"delimiter": delimiter})
        docs = list(islice(loader.lazy_load(), MAX_ROWS))

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    file_type = ext.lstrip(".")
    for d in docs:
        d.metadata["filename"] = filename
        d.metadata["file_type"] = file_type

    chunks = splitter.split_documents(docs)

    # FIX 3: drop tiny junk chunks (but keep short csv/excel rows)
    if file_type in PROSE_TYPES:
        chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]

    if not chunks:
        raise ValueError("No text could be extracted (scanned PDF or empty file?)")

    delete_old_chunks(filename)
    vectorstore.add_documents(chunks)
    return len(chunks)


# ------------------------------------------------------------------
# CHAINS
# ------------------------------------------------------------------

# FIX 1: one clear set of words, and the code below checks the SAME words.
# Also: questions about the user / the conversation are "chat" (no document search).
GREETINGS = {"hi", "hii", "hello", "hey", "yo", "thanks", "thank you",
             "bye", "good morning", "good afternoon", "good evening", "good night"}

classifier = ChatPromptTemplate.from_messages([
    ("system", """Classify the user message into exactly one word:
chat = greeting, thanks, small talk, questions about the bot, or questions about the
       user or this conversation (like "what is my name" or "what did I ask before")
pdf = user clearly asks about the PDF file
docx = user clearly asks about the Word document
txt = user clearly asks about the text files
csv = user clearly asks about the csv file
xlsx = user clearly asks about the Excel file
all = any other question about document content, or unclear

Return only the one word. No explanation."""),
    ("human", "{input}"),
]) | llm | StrOutputParser()

# FIX 2: the user's name is passed into the prompt
chat_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant that answers questions about the user's documents. "
               "The user's name is {user_name}. Reply briefly. "
               "Do not make up facts about the documents."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
]) | llm | StrOutputParser()

# FIX 4: rewrite follow-up questions so "he", "she", "it" become real names
rewrite_chain = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest message as a standalone question using the chat history. "
               "Replace pronouns (he, she, it, they) with the actual names. "
               "If it is already clear, return it unchanged. Return only the question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
]) | llm | StrOutputParser()

rag_chain = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".
Keep answers concise and accurate.

context = {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
]) | llm | StrOutputParser()


# ------------------------------------------------------------------
# MEMORY
# FIX 5: a plain Python list per session (InMemoryChatMessageHistory is deprecated)
# ------------------------------------------------------------------
store = {}   # {session_id: [HumanMessage, AIMessage, ...]}

def keyword_search(query, file_type=None):
    # keep only useful words; every one of them must appear in the chunk
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if w.lower() not in IGNORE_WORDS]
    if not words:
        return []
    and_query = " & ".join(words)

    plain_url = DATABASE_URL.replace("postgresql+psycopg", "postgresql")
    with psycopg.connect(plain_url) as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.document, e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = %(collection)s
              AND to_tsvector('english', e.document) @@ to_tsquery('english', %(q)s)
              AND (%(ftype)s::text IS NULL OR e.cmetadata->>'file_type' = %(ftype)s::text)
            ORDER BY ts_rank(to_tsvector('english', e.document), to_tsquery('english', %(q)s)) DESC
            LIMIT %(limit)s
            """,
            {"collection": COLLECTION_NAME, "q": and_query,
             "ftype": file_type, "limit": CANDIDATES},
        ).fetchall()
        print("keyword query:", and_query, "| hits:", len(rows))

    return [Document(id=str(row[0]), page_content=row[1], metadata=row[2]) for row in rows]


def hybrid_search(query, file_type=None):
    file_filter = {"file_type": {"$eq": file_type}} if file_type else None

    # list 1: search by meaning (weak matches are dropped BEFORE merging)
    vector_found = vectorstore.similarity_search_with_score(
        query, k=CANDIDATES, filter=file_filter)
    print("vector distances:", [round(float(score), 3) for doc, score in vector_found])
    vector_docs = [doc for doc, score in vector_found if score <= SCORE_LIMIT]

    # list 2: search by words
    keyword_docs = keyword_search(query, file_type)

    # Reciprocal Rank Fusion: each list gives a chunk 1/(60 + rank) points
    points = {}      # chunk id -> total points
    docs = {}        # chunk id -> the chunk
    found_by = {}    # chunk id -> which searches found it

    for method, ranked_docs in (("meaning", vector_docs), ("words", keyword_docs)):
        for rank, doc in enumerate(ranked_docs, start=1):
            points[doc.id] = points.get(doc.id, 0) + 1 / (RRF_K + rank)
            docs[doc.id] = doc
            found_by.setdefault(doc.id, []).append(method)

    best_ids = sorted(points, key=points.get, reverse=True)[:TOP_K]
    return [(docs[i], points[i], found_by[i]) for i in best_ids]


# ------------------------------------------------------------------
# ASK
# ------------------------------------------------------------------
def ask(session_id, user_name, question):
    history = store.setdefault(session_id, [])
    recent = history[-6:]    # last 3 exchanges are enough

    clean = question.lower().strip(" !.?,")

    if clean in GREETINGS:
        route = "chat"                       # obvious greeting, skip the classifier
    else:
        raw = classifier.invoke({"input": question})
        print("classifier said:", repr(raw))   # remove once it works
        route = raw.strip().lower().strip(".")
        if route not in ["chat", "all"] + FILE_TYPES:
            route = "all"                    # unexpected answer -> search everything

    sources = []
    search_question = None

    if route == "chat":
        # quick reply, no searching
        answer = chat_chain.invoke({
            "input": question, "chat_history": recent, "user_name": user_name
        })

    else:
        # FIX 4: rewrite only when there is earlier conversation to refer to
        search_question = question
        if recent:
            search_question = rewrite_chain.invoke({"input": question, "chat_history": recent}).strip()

        file_type = route if route in FILE_TYPES else None
        results = hybrid_search(search_question, file_type)

        if not results:
            answer = "I couldn't find anything relevant in the documents."
        else:
            context = "\n\n".join(doc.page_content for doc, points, how in results)
            answer = rag_chain.invoke({
                "input": question, "chat_history": recent, "context": context
            })
            for doc, points, how in results:
                page = doc.metadata.get("page")
                sources.append({
                    "filename": doc.metadata.get("filename", "unknown"),
                    "page": page + 1 if page is not None else None,
                    "score": round(points, 4),
                    "found_by": " + ".join(how),
                    "preview": doc.page_content[:200],
                })

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "route": route,
        "search_question": search_question,
        "sources": sources,
    }