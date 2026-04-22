import fitz
import torch
import uuid
from datetime import datetime
from lib.constants import PDFFOLDER, CHUNKSIZE, OVERLAP
from lib.db import upload_new_doc
from lib.doc_analysis import create_records, get_text, chunkify, encode_doc



if __name__ == '__main__':
    filename = 'webster_dic.pdf'
    with open(f'{PDFFOLDER}/{filename}', 'rb') as f:
        doc = fitz.open(f)
        
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