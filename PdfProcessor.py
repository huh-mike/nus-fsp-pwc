from http.client import responses

import requests
from pdfminer.high_level import extract_text

def download_pdf(url, filename):
    '''
    :param url: url to the pdf
    :return: file path to the downloaded pdf
    '''
    response = None
    final_path = None
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            final_path = f"Download/{filename}.pdf"
            with open(final_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded PDF: 'Download/{filename}'")
    except Exception as e:
        print(f"Error Downloading PDF from: {url}, Error: {e}, status code: {response.status_code}")

    return final_path

def extract_text_from_pdf(pdf_path):
    '''
    :param pdf_path: filepath of the pdf
    :return: extracted text
    '''
    text = extract_text(pdf_path)
    return text


