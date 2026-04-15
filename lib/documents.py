import vecs
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
POOLER_URL = os.getenv('POOLER_URL')



def init_db(dims: int):
    vx = vecs.create_client(POOLER_URL) # type: ignore
    doc = vx.get_or_create_collection(name="embeddings", dimension=dims)
    doc.create_index(measure=vecs.IndexMeasure.cosine_distance)
    return doc

def batch_upsert(collection, records: list, batch_size=500):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        collection.upsert(records=batch, on_conflict='filename')