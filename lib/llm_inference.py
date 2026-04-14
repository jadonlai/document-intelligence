from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from constants import PHI3MODEL, PHI3MODELPATH, PHI3CONTEXT

pipe_path = Path(PHI3MODELPATH)



def get_phi3_pipeline():
    print('Loading phi3 pipeline...')
    tokenizer = AutoTokenizer.from_pretrained(PHI3MODEL)
    model = AutoModelForCausalLM.from_pretrained( 
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.save_pretrained('./models/')
    print('Saved phi3 model...')
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
    print('Finished loading phi3 pipeline...')
    return pipe

def answer_query(query: str, context_chunks: list[str]):
    context = '\n\n-----\n\n'.join(context_chunks)
    prompt = [
        {'role': 'system', 'content': f"""
         {PHI3CONTEXT}
         
         DOCUMENT CONTEXT:
         {context}
         """},
        {'role': 'user', 'content': f'QUESTION: {query}'}
    ]
    pipe = get_phi3_pipeline()
    outputs = pipe(prompt, return_full_text=False)
    return outputs


if __name__ == '__main__':
    get_phi3_pipeline()