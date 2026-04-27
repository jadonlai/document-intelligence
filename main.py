import fitz
import torch
import uuid
from datetime import datetime
from lib.constants import PDFFOLDER, CHUNKSIZE, OVERLAP
from lib.db import upload_new_doc, vec_get_uuid_from_filename, vec_query_from_uuid
from lib.embeddings import create_records, cross_encode_chunks, encode_query, get_text, chunkify, encode_doc
from lib.llm import generate_stream



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
    
def get_top_chunks(query: str, file: str):
    query_embedding = encode_query(query)
    file_uuid = vec_get_uuid_from_filename(file)
    if file_uuid == -1:
        print('Error getting uuid from filename')
        exit(1)
    response = vec_query_from_uuid(file_uuid, query_embedding.tolist())
    chunks = [item['metadata']['chunk'] for item in response.data] # type: ignore
    top_chunks = cross_encode_chunks(query, chunks) # type: ignore
    return top_chunks

if __name__ == '__main__':
    query = 'define fail'
    file = 'webster_dic.pdf'
    chunks = get_top_chunks(query, file)
    generate_stream(query, [chunk[0] for chunk in chunks])