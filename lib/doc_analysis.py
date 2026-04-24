import fitz
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from .constants import EMBEDDINGMODEL, CROSSENCODERMODEL

SENTENCETRANSFORMER = SentenceTransformer(EMBEDDINGMODEL)
CROSSENCODER = CrossEncoder(CROSSENCODERMODEL)



def get_text(doc: fitz.Document) -> str:
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

def chunkify(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def encode_doc(chunks: list[str], batch_size: int = 32) -> torch.Tensor:
    embeddings = SENTENCETRANSFORMER.encode_document(chunks, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError('Embeddings must be a torch.Tensor')

    return embeddings

def encode_query(query: str) -> torch.Tensor:
    embedding = SENTENCETRANSFORMER.encode_query(query, convert_to_tensor=True)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError('Embedding must be a torch.Tensor')
    return embedding

def cross_encode_chunks(query: str, chunks: list[str], k: int = 10) -> list[tuple[str, torch.Tensor]]:
    pairs = [(query, chunk) for chunk in chunks]
    rerank_scores = CROSSENCODER.predict(pairs)
    ranked = sorted(zip(chunks, rerank_scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]

def create_records(uuid: str, chunks: list[str], embeddings: torch.Tensor) -> list[tuple[str, torch.Tensor, dict[str, str | int]]]:
    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        records.append((
            f'uuid_{uuid}_page_{i}',
            embedding.cpu(),
            {
                'page': i,
                'chunk': chunk,
                'uuid': uuid
            }
        ))
    return records