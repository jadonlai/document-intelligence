CHUNKSIZE = 100
OVERLAP = 25
PDFFOLDER = './pdfs'
EMBEDDINGSFOLDER = './embeddings'
EMBEDDINGMODEL = 'all-MiniLM-L6-v2'
SENTENCETRANSFORMEREMBEDDINGSIZE = 384
PHI3MODELPATH = 'models/Phi-3-mini-4k-instruct'
PHI3MODEL = 'microsoft/Phi-3-mini-4k-instruct'
PHI3CONTEXT = """
You are a helpful document analyst. Answer the question using ONLY the document context below.
If the answer is not in the context, say 'I could not find that information in the document.'
Do not make up any information
"""