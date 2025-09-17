def test_upload():
    from app.uploader import upload_pdf_to_database

    # TODO add route to upload
    upload_pdf_to_database(
        text_file="./tests/fixture/bitcoin_es.pdf",
        theme="Bitcoin",
        subtheme="-",
        collection_name="CryptoCurrency",
    )
    upload_pdf_to_database(
        text_file="./tests/fixture/Ethereum.pdf",
        theme="Ethereum",
        subtheme="-",
        collection_name="CryptoCurrency",
    )
