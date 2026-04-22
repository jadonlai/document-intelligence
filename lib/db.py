from postgrest import APIError, APIResponse
from typing import Literal
import vecs
import os
from dotenv import load_dotenv
from supabase import create_client, Client

from lib.constants import SENTENCETRANSFORMEREMBEDDINGSIZE

load_dotenv()

POOLER_URL = os.getenv('POOLER_URL')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError('DIRECT_URL and SUPABASE_KEY must be set in the environment')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



def doc_check_exists(column: Literal['filename', 'uuid'], value: str) -> APIResponse:
    response = supabase.table('documents')\
        .select(column)\
        .eq(column, value)\
        .execute()
    return response

def doc_insert(filename: str, record: dict) -> APIResponse | Literal[-1]:
    existing = doc_check_exists('filename', filename)
    if existing.data:
        print('Document already exists in documents table')
        return -1
    
    response = supabase.table('documents')\
        .insert(record)\
        .execute()
    return response

def doc_delete(column: Literal['filename', 'uuid'], value: str) -> APIResponse:
    response = supabase.table('documents')\
        .delete()\
        .eq(column, value)\
        .execute()
    return response

def vec_init_db(dim: int) -> vecs.Collection:
    vx = vecs.create_client(POOLER_URL) # type: ignore
    chunks = vx.get_or_create_collection(name="chunks", dimension=dim)
    chunks.create_index(measure=vecs.IndexMeasure.cosine_distance)
    return chunks

def vec_batch_upsert(collection: vecs.Collection, records: list, uuid: str, batch_size=500) -> Literal[-1] | None:
    existing = doc_check_exists('uuid', uuid)
    if existing.data:
        print('Embeddings already exists in chunks table')
        return -1
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        collection.upsert(records=batch)
        
def upload_new_doc(filename: str, doc_record: dict, chunks_records: list, batch_size=500) -> Literal[-1] | None:
    response = doc_insert(filename, doc_record)
    if not isinstance(response, APIResponse):
        return -1
    
    chunks = vec_init_db(SENTENCETRANSFORMEREMBEDDINGSIZE)
    empty_uuid = '00000000-0000-0000-0000-000000000000'
    # Enter in empty uuid to successfully insert
    response = vec_batch_upsert(chunks, chunks_records, empty_uuid, batch_size=batch_size)
    if response == -1:
        response = doc_delete('uuid', doc_record['uuid'])
        return -1