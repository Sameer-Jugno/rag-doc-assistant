import os 
import fitz 

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
                    "source" : file, 
                    "page" : page_num+1, 
                    "page_content" : text
                })
    return data


if __name__ == "__main__" : 
    path = "../data/raw"
    data = extract_text_from_pdfs(path) 
    print("Length : ", len(data)) 
    print("\n" , "-" * 50 )
    print("First 10 Docs from pdfs : ", data[:1]) 
    print("\n" , "-" * 50 )
    print("Last 10 Docs from pdfs : ", data[626:]) 

