import numpy as np
from transformers import pipeline
from haber_cekici import haberleri_getir # Bir önceki scriptimizden fonksiyonu import ediyoruz
import datetime

# --- DUYGU ANALİZİ MODELİNİ YÜKLEME ---
# 'pipeline' fonksiyonu, modeli ve gerekli tüm bileşenleri bizim için kolayca kurar.
# Bu model, ilk çalıştırmada Hugging Face'ten indirilecektir (birkaç yüz MB).
print("Duygu analizi modeli yükleniyor... (ilk seferde uzun sürebilir)")
sentiment_pipeline = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")
print("Model başarıyla yüklendi.")
# ------------------------------------


def gunluk_duygu_skorunu_hesapla(baslik_listesi):
    """
    Verilen bir haber başlıkları listesinin ortalama duygu skorunu hesaplar.
    
    Args:
        baslik_listesi (list): String formatında haber başlıkları.
        
    Returns:
        float: -1 (çok negatif) ile +1 (çok pozitif) arasında bir ortalama skor.
    """
    if not baslik_listesi or "bulunamadı" in baslik_listesi[0]:
        return 0.0 # Haber yoksa duygu nötr (0) kabul edilir.

    skorlar = []
    
    # Modelden tahminleri al
    tahminler = sentiment_pipeline(baslik_listesi)
    
    for tahmin in tahminler:
        label = tahmin['label']
        if label == 'positive':
            skorlar.append(1)
        elif label == 'negative':
            skorlar.append(-1)
        else: # neutral
            skorlar.append(0)
            
    # Eğer hiç skor yoksa 0 döndür, varsa ortalamasını al
    if not skorlar:
        return 0.0
    
    return np.mean(skorlar)


if __name__ == "__main__":
    # Test etmek için dünün haberlerini tekrar çekelim
    sirket = "THY"
    tarih = datetime.date.today() - datetime.timedelta(days=1)
    
    print(f"\n'{sirket}' için {tarih.strftime('%Y-%m-%d')} tarihli haberler alınıyor...")
    haberler = haberleri_getir(sirket, tarih)
    
    print("\n--- HABER BAŞLIKLARI ---")
    for baslik in haberler:
        print(f"- {baslik}")
        
    print("\nOrtalama duygu skoru hesaplanıyor...")
    ortalama_skor = gunluk_duygu_skorunu_hesapla(haberler)
    
    print("\n--- SONUÇ ---")
    print(f"Hesaplanan Ortalama Duygu Skoru: {ortalama_skor:.2f}")

    if ortalama_skor > 0.2:
        print("Genel duyarlılık: Pozitif")
    elif ortalama_skor < -0.2:
        print("Genel duyarlılık: Negatif")
    else:
        print("Genel duyarlılık: Nötr")