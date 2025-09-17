def test_api_llamaParse():
    
    import nest_asyncio
    nest_asyncio.apply()
    from llama_parse import LlamaParse
    from app.splitter import text_splitter
    

    api_key="llx-GnR2Ac9d0MsVMo5M8F4RVEQDDKAEaQVAvDc7DN15YjYMc7Sb"
    file_path = "./tests/fixture/algoritmos.pdf"

    parser = LlamaParse(
        api_key=api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="es",  # Optionally you can define a language, default=en
    )

    documents = parser.load_data(file_path=file_path)
    
    text = ""
    doc_text = []
    for doc in documents:
        doc_text.append(doc.text)
    
    doc_text = text_splitter.create_documents(doc_text, metadatas=[{"source":file_path,"theme":"-","subtheme":"-"}])
    chunks=text_splitter.split_documents(documents=doc_text)

    with open('./tests/fixture/output_from_llamaparse.txt', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk.page_content)
    
