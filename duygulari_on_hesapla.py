import pandas as pd
import datetime
from tqdm import tqdm
import os # Dosya varlığını kontrol etmek için os kütüphanesini ekledik

# Diğer script'lerimizden fonksiyonları import ediyoruz
from haber_cekici import haberleri_getir
from duygu_analizi import gunluk_duygu_skorunu_hesapla

# --- PARAMETRELER ---
HİSSE_KODU = "THY"
BASLANGIC_TARIHI = "2020-01-01"
BITIS_TARIHI = "2024-12-31"
CIKTI_DOSYASI = "thy_duygu_skorlari.csv"
# --------------------

if __name__ == "__main__":
    sonuclar = []
    baslangic_gunu = pd.to_datetime(BASLANGIC_TARIHI)

    # --- YENİ EKLENEN KISIM: KAYIT KONTROLÜ VE YÜKLEME ---
    if os.path.exists(CIKTI_DOSYASI):
        print(f"'{CIKTI_DOSYASI}' bulundu. Mevcut ilerleme yükleniyor...")
        df_mevcut = pd.read_csv(CIKTI_DOSYASI)
        # to_dict('records') ile DataFrame'i orijinal 'sonuclar' listesi formatına çeviriyoruz
        sonuclar = df_mevcut.to_dict('records')
        
        # Son kaydedilen tarihten bir sonraki günden devam etmek için başlangıç gününü ayarla
        son_tarih_str = df_mevcut['Date'].iloc[-1]
        baslangic_gunu = pd.to_datetime(son_tarih_str) + datetime.timedelta(days=1)
        print(f"İşleme {baslangic_gunu.strftime('%Y-%m-%d')} tarihinden devam edilecek.")
    else:
        print("Yeni bir hesaplama başlatılıyor...")
    # ----------------------------------------------------

    tarih_araligi = pd.date_range(start=baslangic_gunu, end=BITIS_TARIHI, freq='D')
    
    if tarih_araligi.empty:
        print("Hesaplanacak yeni tarih bulunmuyor. İşlem tamamlanmış.")
    else:
        print(f"'{baslangic_gunu.strftime('%Y-%m-%d')}' ve '{BITIS_TARIHI}' arasındaki günler için duygu skorları hesaplanacak.")
        print("Bu işlem internet hızınıza ve gün sayısına bağlı olarak UZUN sürebilir.")
        
        # tqdm'a başlangıç değerini vererek ilerleme çubuğunun doğru yerden başlamasını sağlıyoruz
        progress_bar = tqdm(tarih_araligi, desc="Duygu Skorları Hesaplanıyor", initial=len(sonuclar), total=len(pd.date_range(start=BASLANGIC_TARIHI, end=BITIS_TARIHI)))

        for gun in progress_bar:
            tarih_str = gun.strftime("%Y-%m-%d")
            
            haberler = haberleri_getir(HİSSE_KODU, gun.date())
            skor = gunluk_duygu_skorunu_hesapla(haberler)
            
            sonuclar.append({'Date': tarih_str, 'sentiment_score': skor})

            # --- YENİ EKLENEN KISIM: PERİYODİK KAYDETME ---
            # Her 20 günde bir ilerlemeyi dosyaya yazarak checkpoint oluşturuyoruz.
            if progress_bar.n % 20 == 0:
                df_ara_kayit = pd.DataFrame(sonuclar)
                df_ara_kayit.to_csv(CIKTI_DOSYASI, index=False)
            # -----------------------------------------------

        # Son kaydı da yap
        df_sonuclar = pd.DataFrame(sonuclar)
        df_sonuclar.to_csv(CIKTI_DOSYASI, index=False)

        print(f"\nİşlem tamamlandı! Sonuçlar '{CIKTI_DOSYASI}' dosyasına kaydedildi.")
        print("Son 5 skor:")
        print(df_sonuclar.tail())