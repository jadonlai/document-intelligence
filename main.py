import fitz
import torch
from sentence_transformers import SentenceTransformer



SENTENCETRANSFORMER = SentenceTransformer("all-MiniLM-L6-v2")



def get_text(doc: fitz.Document):
    text = ""
    for page in doc:
        page_text = page.get_text()
        if isinstance(page_text, str):
            text += page_text
        elif isinstance(page_text, list):
            text += ' '.join(page_text)
        else:
            text += str(page_text)
    return text

def chunkify(text: str, size: int, overlap: int):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def search_embeddings(query, embeddings, top_k=1):
    query_embedding = SENTENCETRANSFORMER.encode_query(query, convert_to_tensor=True)
    scores = []
    similarity_score = SENTENCETRANSFORMER.similarity(query_embedding, embeddings)[0]
    score, i = torch.topk(similarity_score, k=top_k)
    scores.append(score)
    return scores, i



def main():
    doc = fitz.open('webster_dic.pdf')
    text = get_text(doc)
    chunks = chunkify(text, 500, 50)
    print(f'Number of chunks: {len(chunks)}')

    embeddings = SENTENCETRANSFORMER.encode_document(chunks, convert_to_tensor=True)
    print(f'Number of embeddings: {len(embeddings)}')
    
    scores, i = search_embeddings('dog', embeddings)
    for score, index in zip(scores, i):
        print(f'Score: {score.item():.4f}, Chunk: {chunks[index]}')
        
    

if __name__ == '__main__':
    main()
    