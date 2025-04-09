import os
from zipfile import ZipFile
from io import BytesIO
from PIL import Image
import torch
import open_clip
from docx import Document
from llama_index.readers.file.docs import DocxReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extract text using DocxReader, returned as LangChain-compatible Document objects
def extract_text_with_docxreader(docx_path):
    docx_reader = DocxReader()
    documents = docx_reader.load_data(docx_path)
    return [doc.to_langchain_format() for doc in documents]

# Extract images from the DOCX file and convert them into PIL format
def extract_images_as_pil(docx_path):
    images = []
    with ZipFile(docx_path, 'r') as docx_zip:
        image_files = [f for f in docx_zip.namelist() if f.startswith("word/media/")]
        for image_file in image_files:
            image_data = docx_zip.read(image_file)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            images.append(image)
    return images

# load the OpenCLIP model
def load_clip_model(model_path):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', 
        pretrained=model_path, 
        load_weights_only=True
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

# Convert a PIL image to a normalized feature vector using OpenCLIP
def pil_to_vector(image_pil, model, preprocess):
    image = preprocess(image_pil).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)

# Convert tokenized text into normalized feature vectors using OpenCLIP
def text_to_vector(text_list, model, tokenizer):
    tokens = tokenizer(text_list)
    with torch.no_grad(), torch.autocast("cuda"):
        text_features = model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)




if __name__ == "__main__":
    docx_path = "/home/longquan/project/learning/LLM/docs/Spring Bulbs.docx"
    model_path = "/home/longquan/model/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"

    #  Load text and images from DOCX
    documents = extract_text_with_docxreader(docx_path)
    images = extract_images_as_pil(docx_path)

    # spilt text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    text_chunks = splitter.split_documents(documents)
    text_list = [chunk.page_content for chunk in text_chunks]

    # load CLIP model
    model, preprocess, tokenizer = load_clip_model(model_path)

    # convert image to vector
    image_vectors = torch.cat([
        pil_to_vector(img, model, preprocess) for img in images
    ], dim=0)

    # convert text to vector
    text_vectors = text_to_vector(text_list, model, tokenizer)

    print("Image vectors:", image_vectors.shape)
    print("Text vectors:", text_vectors.shape)
