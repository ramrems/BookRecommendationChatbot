import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd
import os
import re
import streamlit as st
import base64
# ─── ENV ─────────────────────────────────────────
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
SUB_DATA_DIR=os.getenv('SUB_DATA_DIR')
background_img = os.path.join(SUB_DATA_DIR, 'library.png')

# Resmi base64'e çevir
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# CSS ile arka plan ayarla
def set_background_image(image_path):
    img_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{img_base64});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Kullanımı
set_background_image(background_img)

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Kitap Öneri Sistemi",
    page_icon="📚",
    layout="wide"
)

@st.cache_data
def load_and_process_data():
    """Veriyi yükle ve işle - cache ile performans artışı"""
    try:
        # ─── EXCEL 1 (Book_Details.csv) ────────────────────────
        df1 = pd.read_csv("data\\Book_Details.csv")
        
        df1_processed = pd.DataFrame({
            "authors": df1["author"],
            "title": df1["book_title"],
            "genres": df1["genres"],
            "description": df1["book_details"],
            "avg_rating": df1["average_rating"],
            "pages": df1["num_pages"],
            "publisher": df1["publication_info"].str.extract(r'^(.*?)(?:,|$)')[0],
            "year": df1["publication_info"].str.extract(r'(\d{4})')[0],
            "image": df1["cover_image_uri"],
            "review_count": df1["num_reviews"],
            "totalratings_count": df1["num_ratings"]
        })
        
        # ─── EXCEL 2 (books.csv) ────────────────────────
        df2 = pd.read_csv("data\\books.csv")
        
        df2_processed = pd.DataFrame({
            "authors": df2["author"],
            "title": df2["title"],
            "genres": df2["genre"],
            "description": df2["desc"],
            "avg_rating": df2["rating"],
            "pages": df2["pages"],
            "publisher": "",  # veri yok
            "year": "",       # veri yok
            "image": df2["img"],
            "review_count": df2["reviews"],
            "totalratings_count": df2["totalratings"]
        })
        
        # ─── VERİLERİ BİRLEŞTİR ─────────────────────────
        combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True)
        return combined_df
        
    except FileNotFoundError as e:
        st.error(f"CSV dosyası bulunamadı: {e}")
        return None
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

@st.cache_resource
def setup_rag_chain():
    """RAG zincirini kur - resource cache ile tek seferlik yükleme"""
    combined_df = load_and_process_data()
    if combined_df is None:
        return None, None
        
    # ─── LangChain Document'larına ÇEVİR ────────────
    docs_from_books = [
        Document(
            page_content=(
                f"Yazar: {row['authors']}\n"
                f"Kitap Adı: {row['title']}\n"
                f"Tür: {row['genres']}\n"
                f"Açıklama: {row['description']}\n"
                f"Ortalama Puan: {row['avg_rating']}\n"
                f"Sayfa Sayısı: {row['pages']}\n"
                f"Yayınevi: {row['publisher']}\n"
                f"Yıl: {row['year']}\n"
                f"Yorum Sayısı: {row['review_count']}\n"
                f"Toplam Puanlayan: {row['totalratings_count']}\n"
                f"Görsel URL: {row['image']}"
            ),
            metadata={
                "source": "excel", 
                "title": row['title'], 
                "author": row['authors'],
                "image": row['image'],
                "rating": row['avg_rating'],
                "genres": row['genres']
            }
        )
        for _, row in combined_df.iterrows()
    ]
    
    # ─── DÖKÜMANLARI BÖL ─────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs_from_books)
    
    # ─── Vektör Veritabanı OLUŞTUR ──────────────────
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Eğer veritabanı mevcutsa yükle, yoksa oluştur
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(
            documents=split_docs, 
            embedding=embeddings, 
            persist_directory="./chroma_db"
        )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=500)
    
    system_prompt = (
      """Sen uzman bir kitap önerici asistanısın. Verilen bağlam bilgilerini kullanarak kullanıcıya en uygun kitap önerilerini sun.

Önerilerini şu formatta ver (görsel URL'si varsa ekle):
1. **Kitap Adı** - Yazar Adı
   - Tür: [tür bilgisi]
   - Puan: [varsa puan]
   - Neden öneriyorum: [kısa açıklama]

Öneriler verirken:
- Kullanıcının istediği tarz/türe uygun kitapları öner
- Benzer kitapları okuyanlara hitap edecek seçenekleri sun
- Her öneri için kısa ama ikna edici açıklama yap
- En fazla 5 kitap öner
- Kullanıcı çok genel bir soru sorduysa o alandaki en popüler kitaplardan cevap ver
- Eğer bağlamda uygun kitap yoksa: "Veritabanımda bu konuda yeterli kitap bulunamadı" de, kendi genel bilgini kullanarak kitap uydurma! genel öneriler sun.
- Görsel URL'si olan kitaplar için mutlaka kitaba ait görseli de arayüzde göster, yoksa bu kitaba ait görseli web'den bul ve ekle arayüze

Bağlam bilgileri:
{context}"""
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain, combined_df

def parse_book_recommendation(text):
    """AI cevabındaki kitap önerilerini parse et ve görsel bilgisini çıkar"""
    books = []
    
    # Kitap önerilerini bul (1. **Kitap Adı** formatında)
    book_pattern = r'\d+\.\s*\*\*(.*?)\*\*\s*-\s*(.*?)(?:\n|$)'
    book_matches = re.findall(book_pattern, text, re.MULTILINE)
    
    for book_title, author in book_matches:
        book_info = {
            'title': book_title.strip(),
            'author': author.strip(),
            'image': None,
            'details': ''
        }
        
        # Bu kitap için detayları bul
        book_section_pattern = rf'\*\*{re.escape(book_title.strip())}\*\*.*?(?=\d+\.\s*\*\*|\Z)'
        book_section = re.search(book_section_pattern, text, re.DOTALL)
        
        if book_section:
            section_text = book_section.group(0)
            book_info['details'] = section_text
            
            # Görsel URL'sini bul
            image_pattern = r'Görsel:\s*(https?://[^\s\n]+)'
            image_match = re.search(image_pattern, section_text)
            if image_match:
                book_info['image'] = image_match.group(1)
        
        books.append(book_info)
    
    return books

def display_book_recommendation(recommendation_text):
    """Kitap önerilerini görsel ile birlikte göster"""
    books = parse_book_recommendation(recommendation_text)
    
    if not books:
        # Eğer parse edilemezse normal metni göster
        st.markdown(recommendation_text)
        return
    
    # Parse edilmiş kitapları göster
    for i, book in enumerate(books, 1):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                print("book image")
                print(book['image'])
                if book['image'] and book['image'].strip():
                    try:
                        st.image(book['image'], width=200, caption=book['title'])
                    except:
                        st.write("🖼️ Görsel yüklenemedi")
                        st.markdown(f"**{book['title']}** - {book['author']}")

            with col2:
                st.markdown(book['details'])
            
            if i < len(books):
                st.divider()

# ─── Ana Uygulama ───────────────────────────────
def main():
    st.title("📚 Kitap Öneri Sistemi")
    
    # API anahtarı kontrolü
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY çevre değişkeni bulunamadı! .env dosyanızı kontrol edin.")
        st.stop()
    
    # RAG zincirini yükle
    with st.spinner("Sistem yükleniyor..."):
        rag_chain, df = setup_rag_chain()
    
    if rag_chain is None:
        st.error("Sistem yüklenemedi. CSV dosyalarınızı kontrol edin.")
        st.stop()
        
    # Örnek sorular
    st.markdown("### 💡 Örnek Sorular:")
    example_questions = [
        "Bilim kurgu türünde hangi kitapları önerirsin?",
        "Yüksek puanlı romantik kitaplar nelerdir?",
        "500 sayfa altındaki kısa kitaplar önerir misin?",
        "Stephen King'in kitapları hakkında bilgi verir misin?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        if cols[i % 2].button(question, key=f"example_{i}"):
            st.session_state.example_query = question
    
    # Chat arayüzü
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Geçmiş mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "content" in message:
                display_book_recommendation(message["content"])
            else:
                st.markdown(message["content"])
    
    # Yeni mesaj
    query = st.chat_input("Kitap hakkında bir şey sorun...")
    
    # Örnek soru tıklandıysa
    if "example_query" in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    
    if query:
        # Kullanıcı mesajını göster
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # AI cevabını al ve göster
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response["answer"]
                    display_book_recommendation(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Üzgünüm, bir hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()