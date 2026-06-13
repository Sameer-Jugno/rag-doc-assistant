import os 
import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter 


def extract_text_from_pdfs(path) : 
        
    if not os.path.exists(path) : 
        raise FileNotFoundError(f"The Director {path} not exists.")

    data = []
    for file in os.listdir(path) : 
        if file.endswith(".pdf"):
            doc = fitz.open(f"{path}/{file}") 
            for page_num, page in enumerate(doc) : 
                text = page.get_text().strip() 
                
                data.append({
                    "metadata" : {
                                "source" : file, 
                                "page" : page_num+1, 
                                "company" : file.split("-")[0]
                            },
                    "page_content" : text
                })
            # company_name = file.split("-")[0]
            # print(f"company : {company_name}")
    return data

def chunk_documents(data) : 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap = 200
    )
    texts = [txt['page_content'] for txt in data] 
    meta_datas = [dt['metadata'] for dt in data]

    chunks = splitter.create_documents(texts=texts, metadatas=meta_datas)
    return chunks 


if __name__ == "__main__" : 

    # Step 1 : Text Extraction 
    path = "../data/raw"
    data = extract_text_from_pdfs(path) 
    print("\n\n" , "-" * 100 )
    print("Length : ", len(data)) 
    print("First Doc from pdfs : ", data[:1]) 


    # Step 2 : Chunking the Text
    chunks = chunk_documents(data)
    print("\n\n" , "-" * 100 )
    print("Length : ", len(data)) 
    print("Chunk_1 : ", chunks[0]) 
