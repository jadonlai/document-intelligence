import fitz
from pathlib import Path
from lib.constants import EMBEDDINGSPATH
from lib.doc_analysis import chunkify, encode_doc, get_text, save_embeddings, open_embeddings, search_embeddings
from lib.llm_inference import answer_query

embeddings_path = Path(EMBEDDINGSPATH)



def main():
    doc = fitz.open('webster_dic.pdf')
    text = get_text(doc)
    if embeddings_path.is_file():
        chunks, embeddings = open_embeddings(EMBEDDINGSPATH)
    else:
        chunks = chunkify(text, 100, 25)
        embeddings = encode_doc(chunks)
        save_embeddings(chunks, embeddings) # type: ignore
    
    # scores, i = search_embeddings('What is a dog?', embeddings) # type: ignore
    # for score, index in zip(scores, i):
    #     print(f'Score: {score.item():.4f}, Chunk: {chunks[index]}')
        
    answer = answer_query(
        query='What is a dog?',
        context_chunks=chunks[50:55]
    )
    print(answer)
    
    

if __name__ == '__main__':
    main()
    