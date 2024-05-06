def test_upload():
    from app.uploader import upload_pdf_to_database

    upload_pdf_to_database(
        text_file="chainlit_app/tests/pdfs_prueba/bitcoin_es.pdf",
        theme="Bitcoin",
        subtheme="-",
        collection_name="CryptoCurrency",
    )
    upload_pdf_to_database(
        text_file="chainlit_app/tests/pdfs_prueba/Ethereum.pdf",
        theme="Ethereum",
        subtheme="-",
        collection_name="CryptoCurrency",
    )
