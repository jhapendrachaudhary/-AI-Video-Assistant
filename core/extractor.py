from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )

def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{{text}}"),
        ])
        | llm
        | StrOutputParser()
    )

def extract_action_items(transcript: str, output_language: str = "english") -> str:
    lang_instruction = f" in {output_language}" if output_language.lower() != "english" else ""
    system = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        f"Format as a numbered list{lang_instruction}. If none found say 'No action items found.'"
    )
    chain = build_chain(system)
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str, output_language: str = "english") -> str:
    lang_instruction = f" in {output_language}" if output_language.lower() != "english" else ""
    system = (
        "You are an expert meeting analyst. From the meeting transcript, "
        f"extract all key decisions made. Format as a numbered list{lang_instruction}. "
        "If none found say 'No key decisions found.'"
    )
    chain = build_chain(system)
    return chain.invoke(transcript)

def extract_questions(transcript: str, output_language: str = "english") -> str:
    lang_instruction = f" in {output_language}" if output_language.lower() != "english" else ""
    system = (
        "From the meeting transcript, extract all unresolved questions "
        f"or topics needing follow-up. Format as a numbered list{lang_instruction}. "
        "If none found say 'No open questions found.'"
    )
    chain = build_chain(system)
    return chain.invoke(transcript)