from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from dotenv import load_dotenv
import os

# ─── ENV ─────────────────────────────────────────
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
SUB_DATA_DIR=os.getenv('SUB_DATA_DIR')
csv_path1 = os.path.join(BASE_DIR, SUB_DATA_DIR, 'Book_Details.csv')
csv_path2 = os.path.join(BASE_DIR, SUB_DATA_DIR, 'books.csv')

# ─── EXCEL 1 ─────────────────────────────────────
df1 = pd.read_csv(csv_path1)
df1_processed = pd.DataFrame({
    "authors": df1["author"],
    "title": df1["book_title"],
    "genres": df1["genres"],
    "description": df1["book_details"],
    "avg_rating": df1["average_rating"],
    "pages": df1["num_pages"],
    "publisher": df1["publication_info"].str.extract(r'^(.*?)(?:,|$)')[0],
    "year": df1["publication_info"].str.extract(r'(\d{4})')[0],
    "image": df1["cover_image_uri"].fillna('').astype(str),  
    "review_count": df1["num_reviews"],
    "totalratings_count": df1["num_ratings"]
})

# ─── EXCEL 2 ─────────────────────────────────────
df2 = pd.read_csv(csv_path2)
df2_processed = pd.DataFrame({
    "authors": df2["author"],
    "title": df2["title"],
    "genres": df2["genre"],
    "description": df2["desc"],
    "avg_rating": df2["rating"],
    "pages": df2["pages"],
    "publisher": "",  # veri yok
    "year": "",       # veri yok
    "image": df2["img"].fillna('').astype(str),
    "review_count": df2["reviews"],
    "totalratings_count": df2["totalratings"]
})

# ─── VERİLERİ BİRLEŞTİR ─────────────────────────
combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True)

# ─── DÖKÜMANLARA ÇEVİR ──────────────────────────
docs_from_books = [
    Document(
        page_content=(
            f"Yazar: {str(row['authors'])}\n"
            f"Kitap Adı: {str(row['title'])}\n"
            f"Tür: {str(row['genres'])}\n"
            f"Açıklama: {str(row['description'])}\n"
            f"Ortalama Puan: {str(row['avg_rating'])}\n"
            f"Sayfa Sayısı: {str(row['pages'])}\n"
            f"Yayınevi: {str(row['publisher'])}\n"
            f"Yıl: {str(row['year'])}\n"
            f"Yorum Sayısı: {str(row['review_count'])}\n"
            f"Toplam Puanlayan: {str(row['totalratings_count'])}"
            f"image: {str(row['image'])}"
        ),
        metadata={"source": "excel"}
    )
    for _, row in combined_df.iterrows()
]

# ─── DÖKÜMANLARI BÖL ─────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
split_docs = text_splitter.split_documents(docs_from_books)

# ─── Vektör Veritabanı OLUŞTUR ──────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.persist()

print("✅ Vektör veritabanı başarıyla oluşturuldu ve ./chroma_db dizinine kaydedildi.")
