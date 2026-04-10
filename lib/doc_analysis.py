import fitz
import torch
import pickle
from sentence_transformers import SentenceTransformer
from constants import EMBEDDINGSPATH, EMBEDDINGMODEL

SENTENCETRANSFORMER = SentenceTransformer(EMBEDDINGMODEL)



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
    print('Chunkifying...')
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        chunks.append(chunk)
    print(f'Finished chunkifying with {len(chunks)} chunks')
    return chunks

def save_embeddings(chunks: list[str], embeddings: torch.Tensor):
    print('Saving embeddings...')
    with open(EMBEDDINGSPATH, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
    
def open_embeddings(path: str):
    print('Loading embeddings...')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    chunks = data['chunks']
    embeddings = data['embeddings']
    return chunks, embeddings

def encode_doc(chunks: list[str]):
    print('Encoding document...')
    embeddings = SENTENCETRANSFORMER.encode_document(chunks, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    print(f'Finished encoding with {len(embeddings)} embeddings')
    return embeddings

def search_embeddings(query: str, embeddings: torch.Tensor, top_k=1):
    query_embedding = SENTENCETRANSFORMER.encode_query(query, convert_to_tensor=True)
    scores = []
    similarity_score = SENTENCETRANSFORMER.similarity(query_embedding, embeddings)[0] # type: ignore
    score, i = torch.topk(similarity_score, k=top_k)
    scores.append(score)
    return scores, i