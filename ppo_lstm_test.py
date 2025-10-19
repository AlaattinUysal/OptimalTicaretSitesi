import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from veri_hazirlama import veri_cek_ve_hazirla
from ticaret_ortami import TicaretOrtami

MODEL_DOSYASI = "ppo_lstm_model.zip" # Yeni model dosyamızın adı
HİSSE_KODU = "THYAO.IS"
BASLANGIC_BAKIYE = 100000

TEST_BASLANGIC_TARIHI = "2025-01-01"
TEST_BITIS_TARIHI = "2025-09-07"

print("Test verisi çekiliyor...")
test_verisi = veri_cek_ve_hazirla(
    hisse_kodu=HİSSE_KODU,
    baslangic_tarihi=TEST_BASLANGIC_TARIHI,
    bitis_tarihi=TEST_BITIS_TARIHI
)

if test_verisi is not None and len(test_verisi) > 15:
    # LSTM'li ortamı kuruyoruz
    env = TicaretOrtami(df=test_verisi, baslangic_bakiye=BASLANGIC_BAKIYE)

    model = PPO.load(MODEL_DOSYASI)
    print(f"'{MODEL_DOSYASI}' başarıyla yüklendi.")

    print("Test simülasyonu başlatılıyor...")
    obs, info = env.reset()
    done = False
    aksiyon_gecmisi = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        aksiyon_gecmisi.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Test simülasyonu tamamlandı.")

    ppo_sonuc = info['toplam_portfoy_degeri']
    ilk_fiyat = test_verisi['Close'].iloc[0]
    son_fiyat = test_verisi['Close'].iloc[-1]
    al_ve_tut_sonuc = (BASLANGIC_BAKIYE / ilk_fiyat) * son_fiyat

    print("\n--- PPO (LSTM) TEST SONUÇLARI ---")
    print(f"Başlangıç Bakiyesi: {BASLANGIC_BAKIYE:,.2f} TL")
    print("-" * 30)
    print(f"PPO Ajanının Portföy Sonucu: {ppo_sonuc:,.2f} TL")
    print(f"'Al ve Tut' Stratejisi Sonucu: {al_ve_tut_sonuc:,.2f} TL")
    print("-" * 30)
    ppo_getiri = ((ppo_sonuc / BASLANGIC_BAKIYE) - 1) * 100
    al_ve_tut_getiri = ((al_ve_tut_sonuc / BASLANGIC_BAKIYE) - 1) * 100
    print(f"PPO Ajanı Getirisi: %{ppo_getiri:.2f}")
    print(f"'Al ve Tut' Getirisi: %{al_ve_tut_getiri:.2f}")
    if ppo_getiri > al_ve_tut_getiri:
        print("\nSonuç: PPO (LSTM) ajanı, 'Al ve Tut' stratejisinden daha iyi performans gösterdi. ✅")
    else:
        print("\nSonuç: PPO (LSTM) ajanı, 'Al ve Tut' stratejisini geçemedi. ❌")

    plt.figure(figsize=(16, 8))
    plt.plot(test_verisi.index, test_verisi['Close'], label='THYAO Fiyat', color='blue', alpha=0.6)
    baslangic_indeksi = env.sequence_length - 1
    alim_noktalari = [i + baslangic_indeksi for i, a in enumerate(aksiyon_gecmisi) if a == 0]
    plt.plot(test_verisi.index[alim_noktalari], test_verisi['Close'].iloc[alim_noktalari], '^', markersize=10, color='green', label='Alım Yapıldı')
    satim_noktalari = [i + baslangic_indeksi for i, a in enumerate(aksiyon_gecmisi) if a == 1]
    plt.plot(test_verisi.index[satim_noktalari], test_verisi['Close'].iloc[satim_noktalari], 'v', markersize=10, color='red', label='Satım Yapıldı')
    plt.title('PPO Ajanının (LSTM) Test Verisi Üzerindeki Alım-Satım Noktaları')
    plt.xlabel('Tarih')
    plt.ylabel('Hisse Fiyatı (TL)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Test için yeterli veri bulunamadı.")