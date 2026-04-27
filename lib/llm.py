import ollama
from lib.constants import LLMMODEL, LLMCONTEXT



def generate_stream(prompt: str, chunks: list[str]) -> None:
    context = '\n\n'.join(f'[Chunk {i+1}]:\n{chunk}' for i, chunk in enumerate(chunks))
    system_prompt = f'''
        {LLMCONTEXT}

        Document Excerpts:
        {context}
    '''
    
    stream = ollama.chat(
        model=LLMMODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        stream=True
    )
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print()
