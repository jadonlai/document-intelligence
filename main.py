import fitz
from sentence_transformers import CrossEncoder
import torch
import uuid
from datetime import datetime
from lib.constants import PDFFOLDER, CHUNKSIZE, OVERLAP
from lib.db import doc_check_exists, doc_insert, upload_new_doc, vec_get_uuid_from_filename, vec_query_from_uuid
from lib.doc_analysis import create_records, encode_query, get_text, chunkify, encode_doc

CROSSENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')



def open_file(filename: str) -> fitz.Document:
    with open(f'{PDFFOLDER}/{filename}', 'rb') as f:
        doc = fitz.open(f)
    return doc

def upload_doc_to_db() -> None:
    filename = 'webster_dic.pdf'
    doc = open_file(filename)
        
    print('Extracting text...')
    text = get_text(doc)
    
    print('Splitting into chunks...')
    chunks = chunkify(text, CHUNKSIZE, OVERLAP)
    
    print('Encoding document...')
    embeddings = encode_doc(chunks, batch_size=64)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError('Embeddings must be a torch.Tensor')
    
    print('Creating records...')
    unique_uuid = str(uuid.uuid4())
    doc_record = {
        'filename': filename,
        'created_at': str(datetime.now()),
        'uuid': unique_uuid
    }
    
    print('Uploading document to database...')
    response = upload_new_doc(filename, doc_record, create_records(unique_uuid, chunks, embeddings))
    if response == -1:
        print('Error uploading document to database')
        exit(1)
    
    print('Document uploaded to database successfully')

if __name__ == '__main__':
    # filename = 'webster_dic.pdf'
    # doc = open_file(filename)
    # text = get_text(doc)
    # chunks = chunkify(text, CHUNKSIZE, OVERLAP)
    # embeddings = encode_doc(chunks)
    # scores, i = search_embeddings('canine dog', embeddings)
    # print(scores, i)
    
    query = 'what the dog doing'
    query_embedding = encode_query(query)
    file_uuid = vec_get_uuid_from_filename('webster_dic.pdf')
    if file_uuid == -1:
        print('Error getting uuid from filename')
        exit(1)
    response = vec_query_from_uuid(file_uuid, query_embedding.tolist())
    candidates = [item['metadata']['chunk'] for item in response.data] # type: ignore
        
    pairs = [(query, chunk) for chunk in candidates]
    rerank_scores = CROSSENCODER.predict(pairs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

    top_results = [x[0] for x in ranked[:10]]
    for x in top_results:
        print(x + '\n\n')
