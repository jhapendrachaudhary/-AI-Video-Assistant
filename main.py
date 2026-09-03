from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source: str, input_language: str = "english", output_language: str = "english") -> dict:
    print("Starting AI Video Assistant")
    chunks = process_input(source)
    transcript = transcribe_all(chunks, input_language)
    print(f"Raw transcription (first 300 chars): {transcript[:300]}")

    title = generate_title(transcript, output_language)
    summary = summarize(transcript, output_language)
    action_items = extract_action_items(transcript, output_language)
    decisions = extract_key_decisions(transcript, output_language)
    questions = extract_questions(transcript, output_language)

    rag_chain = build_rag_chain(transcript, output_language)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    input_language = input("Input language (english, hinglish, nepali, hindi, etc.): ").strip() or "english"
    output_language = input("Output language for summary/chat (english or nepali): ").strip() or "english"
    result = run_pipeline(source, input_language, output_language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")