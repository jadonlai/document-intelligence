import fitz
import torch
import pickle
from sentence_transformers import SentenceTransformer
from .constants import CHUNKSIZE, EMBEDDINGMODEL, OVERLAP, PDFFOLDER, EMBEDDINGSFOLDER

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
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def save_embeddings(path: str, chunks: list[str], embeddings: torch.Tensor):
    with open(path, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
        
def save_embeddings_from_text(filename: str):
    doc = fitz.open(f'{PDFFOLDER}/{filename}')
    text = get_text(doc)
    chunks = chunkify(text, CHUNKSIZE, OVERLAP)
    embeddings = encode_doc(chunks)
    save_embeddings(f'{EMBEDDINGSFOLDER}/{filename[:-4]}.pkl', chunks, embeddings) # type: ignore
    
def open_embeddings(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    chunks = data['chunks']
    embeddings = data['embeddings']
    return chunks, embeddings

def encode_doc(chunks: list[str]):
    embeddings = SENTENCETRANSFORMER.encode_document(chunks, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    return embeddings

def search_embeddings(query: str, embeddings: torch.Tensor, top_k=1):
    query_embedding = SENTENCETRANSFORMER.encode_query(query, convert_to_tensor=True)
    scores = []
    similarity_score = SENTENCETRANSFORMER.similarity(query_embedding, embeddings)[0] # type: ignore
    score, i = torch.topk(similarity_score, k=top_k)
    scores.append(score)
    return scores, i

def create_records(filename: str, chunks: list[str], embeddings: torch.Tensor):
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        records.append((
            f'chunk_{i}',
            embedding,
            {
                'filename': filename,
                'page': i,
                'chunk': chunk
            }
        ))
    return records