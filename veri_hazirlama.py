import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def veri_cek_ve_hazirla(hisse_kodu, baslangic_tarihi, bitis_tarihi, duygu_dosyasi="thy_duygu_skorlari.csv"):
    try:
        print(f"1. Adım: Ana hisse senedi ({hisse_kodu}) verisi çekiliyor...")
        df_hisse = yf.download(hisse_kodu, start=baslangic_tarihi, end=bitis_tarihi)
        if df_hisse.empty or 'Close' not in df_hisse.columns:
            print(f"HATA: {hisse_kodu} için geçerli veri ('Close' sütunu) bulunamadı.")
            return None
        
        # --- DÜZELTME: Doğru katmanı (Ticker adı) atıyoruz ---
        if isinstance(df_hisse.columns, pd.MultiIndex):
            df_hisse.columns = df_hisse.columns.droplevel(1) 
        # ---------------------------------------------------
        print("-> Başarılı.")

        print("2. Adım: Endeks verisi (XU100.IS) çekiliyor...")
        df_endeks = yf.download("XU100.IS", start=baslangic_tarihi, end=bitis_tarihi)
        if df_endeks.empty or 'Close' not in df_endeks.columns:
            print("HATA: XU100.IS için geçerli veri ('Close' sütunu) bulunamadı.")
            return None
            
        # --- DÜZELTME: Doğru katmanı (Ticker adı) atıyoruz ---
        if isinstance(df_endeks.columns, pd.MultiIndex):
            df_endeks.columns = df_endeks.columns.droplevel(1)
        # ---------------------------------------------------
        print("-> Başarılı.")
        
        df_endeks['bist100_getiri'] = df_endeks['Close'].pct_change()
        
        print(f"3. Adım: '{duygu_dosyasi}' dosyasındaki duygu verisi okunuyor...")
        df_duygu = pd.read_csv(duygu_dosyasi)
        
        df_hisse.index = pd.to_datetime(df_hisse.index)
        df_duygu['Date'] = pd.to_datetime(df_duygu['Date'])
        df_endeks.index = pd.to_datetime(df_endeks.index)

        print("4. Adım: Tüm veriler birleştiriliyor...")
        veri_birlesik = pd.merge(df_hisse, df_duygu, left_index=True, right_on='Date', how='left').set_index('Date')
        veri_birlesik = pd.merge(veri_birlesik, df_endeks[['bist100_getiri']], left_index=True, right_index=True, how='left')
        
        veri_birlesik['sentiment_score'].fillna(0, inplace=True)
        veri_birlesik['bist100_getiri'].fillna(0, inplace=True)

        print("5. Adım: Teknik göstergeler hesaplanıyor...")
        veri_birlesik['SMA_14'] = veri_birlesik['Close'].rolling(window=14).mean()
        veri_birlesik['RSI_14'] = _calculate_rsi(veri_birlesik['Close'])
        
        veri = veri_birlesik[['Close', 'SMA_14', 'RSI_14', 'sentiment_score', 'bist100_getiri']].copy()
        veri.dropna(inplace=True)
        
        print(f"\n{hisse_kodu} için tüm veriler başarıyla birleştirildi.")
        print("Son 5 satır:")
        print(veri.tail())
        
        return veri
    except Exception as e:
        print(f"Veri hazırlama sırasında beklenmedik bir hata oluştu: {e}")
        return None

def veriyi_gorsellestir(df, hisse_kodu):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'])
    plt.title(f'{hisse_kodu} Kapanış Fiyatı Geçmişi')
    plt.xlabel('Tarih')
    plt.ylabel('Kapanış Fiyatı (TL)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    veri_cek_ve_hazirla(hisse_kodu="THYAO.IS", baslangic_tarihi="2020-01-01", bitis_tarihi="2024-12-31")