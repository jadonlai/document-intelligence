CHUNKSIZE = 250
OVERLAP = 50
PDFFOLDER = './pdfs'
EMBEDDINGMODEL = 'all-MiniLM-L6-v2'
CROSSENCODERMODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SENTENCETRANSFORMEREMBEDDINGSIZE = 384
LLMMODEL = 'qwen2.5:3b'
LLMCONTEXT = """
You are a helpful document analyst. Answer the question using ONLY the document context below.
If the answer is not in the context, say 'I could not find that information in the document.'
Do not make up any information
"""