import streamlit as st
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = genai.list_models()
for model in models:
    print(model.name, model.supported_generation_methods)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error("GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)


def remove_images_from_pdf(input_pdf):
    output_pdf = BytesIO()
    writer = PdfWriter()

    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                packet = BytesIO()
                can = canvas.Canvas(packet, pagesize=letter)
                lines = text.split('\n')
                y = 750  # Start position for drawing text
                for line in lines:
                    can.drawString(10, y, line)
                    y -= 12  # Move to the next line (12 points lower)
                    if y < 40:  # If we reach the bottom of the page, start a new page
                        can.showPage()
                        y = 750
                can.save()

                packet.seek(0)
                new_pdf = PdfReader(packet)
                writer.add_page(new_pdf.pages[0])

    writer.write(output_pdf)
    output_pdf.seek(0)
    return output_pdf


def get_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            width, height = page.width, page.height
            exclude_footer_region = (0, height - 50, width, height)
            page_text = page.filter(lambda obj: obj["top"] < exclude_footer_region[1]).extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    just say "answer is not available in the context". Do not make up an answer.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    # model = ChatGoogleGenerativeAI(model="models/gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", temperature=0.2)


    chain = (
        {
            "context": lambda x: "\n\n".join([doc.page_content for doc in x["documents"]]),
            "question": lambda x: x["question"],
        }
        | prompt
        | model
    )

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the FAISS index from the local storage
    # Ensure the index is saved in the same directory as this script
    if not os.path.exists("faiss_index"):
        st.error("FAISS index not found. Please process the PDF files first.")
        return
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"documents": docs, "question": user_question})
    st.write("Reply:", response.content)


def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat With PDF")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    text_only_pdfs = [remove_images_from_pdf(pdf) for pdf in pdf_docs]
                    raw_text = ""
                    for pdf in text_only_pdfs:
                        raw_text += get_pdf_text(pdf)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
