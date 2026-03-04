import fitz



def get_text(doc: fitz.Document):
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

def chunkify(text: str, size: int, overlap: int):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        chunks.append(chunk)
    return chunks

def main():
    doc = fitz.open('sample.pdf')
    text = get_text(doc)
    chunks = chunkify(text, 500, 50)
    for i, chunk in enumerate(chunks):
        print('------------' + str(i) + '------------')
        print(chunk)
        
    

if __name__ == '__main__':
    main()