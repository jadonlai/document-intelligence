import fitz
import torch
from sentence_transformers import SentenceTransformer
from .constants import EMBEDDINGMODEL

SENTENCETRANSFORMER = SentenceTransformer(EMBEDDINGMODEL)



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

def search_embeddings(query: str, embeddings: torch.Tensor, top_k=1) -> tuple[list[torch.Tensor], torch.Tensor]:
    query_embedding = SENTENCETRANSFORMER.encode_query(query, convert_to_tensor=True)
    scores = []
    similarity_score = SENTENCETRANSFORMER.similarity(query_embedding, embeddings)[0] # type: ignore
    score, i = torch.topk(similarity_score, k=top_k)
    scores.append(score)
    return scores, i

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