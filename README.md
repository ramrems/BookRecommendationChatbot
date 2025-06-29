# 📚 RAG Tabanlı Kitap Öneri Sistemi

Bu proje, **RAG (Retrieval-Augmented Generation)** teknolojisini kullanarak geliştirilmiş akıllı bir kitap öneri chatbotu'dur. Sistem, 280.000+ kitap verisini analiz ederek kullanıcılara kişiselleştirilmiş kitap önerileri sunar.

## ✨ Özellikler

- 🤖 **AI Destekli Sohbet**: Doğal dil işleme ile kitap önerileri
- 📊 **Kapsamlı Veri**: 280.000+ kitap verisi (2 farklı dataset birleşimi)
- 🔍 **Akıllı Arama**: Vektör tabanlı benzerlik araması

## 🛠️ Teknolojiler

- **Frontend**: Streamlit
- **AI/ML**: Google Gemini (LLM), LangChain
- **Vektör DB**: Chroma
- **Embedding**: Google Generative AI Embeddings
- **Veri İşleme**: Pandas
- **Diğer**: Python, dotenv

## 📊 Veri Seti

Proje iki farklı Kaggle dataset'ini birleştirerek oluşturulmuştur:

1. **Book_Details.csv** - 177.411 kitap
2. **books.csv** - 106.304 kitap

**Toplam**: ~280.000 kitap verisi

### Veri Alanları:
- Yazar bilgisi
- Kitap adı ve açıklaması
- Tür/kategori
- Ortalama puan ve değerlendirme sayısı
- Sayfa sayısı
- Yayınevi ve yayın yılı
- Kapak görseli URL'si

## 🚀 Kurulum

### Gereksinimler

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install langchain-chroma
pip install pandas
pip install python-dotenv
```

### Adım 1: Repository'yi klonlayın

```bash
git clone [repository-url]
cd kitap-oneri-sistemi
```

### Adım 2: Çevre değişkenlerini ayarlayın

`.env` dosyası oluşturun:

```env
GOOGLE_API_KEY=your_google_api_key_here
BASE_DIR=./
SUB_DATA_DIR=data
```

### Adım 3: Veri dosyalarını yerleştirin

```
data/
├── Book_Details.csv
├── books.csv
└── library.png (arka plan görseli)
```

### Adım 4: Vektör veritabanını oluşturun

```bash
python data_processor.py
```

Bu işlem birkaç dakika sürebilir. Tamamlandığında `./chroma_db` klasörü oluşturulacaktır.

### Adım 5: Uygulamayı çalıştırın

```bash
streamlit run app.py
```

## 📁 Proje Yapısı

```
kitap-oneri-sistemi/
│
├── app.py
│   ├── streamlit_app.py  #Ana streamlit uygulaması  
├── .env                   # Çevre değişkenleri
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
├── data/                 # Veri dosyaları
│   ├── Book_Details.csv
│   ├── books.csv
│   └── library.png
│── model/        
│   ├── gemini_model.py    # Veri işleme ve embedding oluşturma   
└── chroma_db/            # Vektör veritabanı (otomatik oluşur)
    └── [chroma files]
```

## 🎯 Kullanım

1. **Uygulama Başlatma**: `streamlit run app.py`
2. **Soru Sorma**: Chat arayüzünden kitap tercihleri hakkında soru sorun
3. **Örnek Sorular**: Hazır örnek sorulardan birini seçin
4. **Sonuçları İnceleme**: AI'dan gelen öneriler görsellerle birlikte görüntülenecek

### Örnek Sorular:

- "Bilim kurgu türünde hangi kitapları önerirsin?"
- "Yüksek puanlı romantik kitaplar nelerdir?"
- "500 sayfa altındaki kısa kitaplar önerir misin?"
- "Stephen King'in kitapları hakkında bilgi verir misin?"
  
![Ekran görüntüsü 2025-06-28 214231](https://github.com/user-attachments/assets/2072886e-35c5-4d04-b777-2ce979c4e395)

## 🔧 Teknik Detaylar

### RAG Mimarisi

1. **Veri Hazırlığı**: CSV dosyalarını birleştirme ve temizleme
2. **Embedding**: Google Generative AI ile vektörel temsil
3. **Vektör Depolama**: Chroma DB ile kalıcı saklama
4. **Retrieval**: Benzerlik tabanlı arama (k=5)
5. **Generation**: Gemini 1.5 Flash ile öneriler

# Model Karşılaştırması
Proje geliştirme sürecinde farklı LLM modelleri test edilmiş ancak API hataları nedeniyle karşılaştırma çalışması tamamlanamamıştır. Mevcut implementasyonda Google Gemini 1.5 Flash modeli kullanılmaktadır.
# ⚠️ Notlar

Excel dosyalarından biri boyut kısıtlamaları nedeniyle repository'de bulunmamaktadır
Model karşılaştırması API erişim sorunları nedeniyle tamamlanamamıştır
Sistem şu anda yalnızca Google Gemini API'si ile çalışmaktadır
