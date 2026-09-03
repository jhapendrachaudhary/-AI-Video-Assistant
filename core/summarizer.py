from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)

def summarize(transcript: str, output_language: str = "english") -> str:
    llm = get_llm()
    lang_instruction = f" in {output_language}" if output_language.lower() != "english" else ""

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Summarize this portion of a meeting transcript concisely{lang_instruction}."),
        ("human", "{{text}}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are an expert meeting summarizer. Combine these partial summaries "
            f"into one final professional meeting summary in bullet points{lang_instruction}."
        ),
        ("human", "{{text}}"),
    ])
    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )
    return combined_chain.invoke(combined)

def generate_title(transcript: str, output_language: str = "english") -> str:
    llm = get_llm()
    lang_instruction = f" in {output_language}" if output_language.lower() != "english" else ""
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                f"Based on the meeting transcript, generate a short professional meeting title "
                f"(max 8 words){lang_instruction}. Only return the title, nothing else."
            ),
            ("human", "{{text}}"),
        ])
        | llm
        | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])