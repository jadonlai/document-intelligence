import vecs
import os

DATABASE_URL = os.getenv('DATABASE_URL')



# create vector store client
vx = vecs.create_client(DATABASE_URL) # type: ignore

docs = vx.get_or_create_collection(name="docs", dimension=3)

# add records to the collection
docs.upsert(
    records=[
        (
         "vec0",           # the vector's identifier
         [0.1, 0.2, 0.3],  # the vector. list or np.array
         {"year": 1973}    # associated  metadata
        ),
        (
         "vec1",
         [0.7, 0.8, 0.9],
         {"year": 2012}
        )
    ]
)

docs.create_index(measure=vecs.IndexMeasure.cosine_distance)

query_res = docs.query(
    data=[0.4,0.5,0.6],  # required
    limit=5,                     # number of records to return
    filters={},                  # metadata filters
    measure="cosine_distance",   # distance measure to use
    include_value=False,         # should distance measure values be returned?
    include_metadata=False,      # should record metadata be returned?
)

print(query_res)