def generate_answer(question: str, contexts: List[dict], model: str = "gpt-3.5-turbo") -> str:
    if client is None:
        raise ValueError("❌ OpenAI client is not initialized. Set 'client = OpenAI(api_key=...)' in app.py before calling this function.")

    assembled = ""
    for i, c in enumerate(contexts, 1):
        assembled += f"Source {i} ({c.get('source', 'unknown')}):\n{c['text']}\n\n"

    prompt = f"""
You are a helpful study assistant. Use ONLY the provided sources to answer.
If the answer is not in the sources, say "I cannot find it in the documents."

Sources:
{assembled}

Question: {question}
Answer concisely and cite the source numbers when possible.
"""

    response = client.chat.completions.create(
        model=model,  # Use gpt-3.5-turbo or gpt-4o-mini
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0
    )

    return response.choices[0].message.content
