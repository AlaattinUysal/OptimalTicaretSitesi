import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Diğer dosyalarımızdan gerekli sınıfları ve fonksiyonları import ediyoruz
from veri_hazirlama import veri_cek_ve_hazirla
from ticaret_ortami import TicaretOrtami
from ajan_ve_egitim import DQNAjan

# --- TEST PARAMETRELERİ ---
MODEL_AGIRLIKLARI_DOSYASI = "dqn_model.weights.h5"
HİSSE_KODU = "THYAO.IS"
BASLANGIC_BAKIYE = 100000

# Eğitim verisi 2024 sonunda bittiği için, test verisini hiç görmediği bir yerden başlatıyoruz.
TEST_BASLANGIC_TARIHI = "2025-01-01"
TEST_BITIS_TARIHI = "2025-09-05"  # Bugünün tarihi

# --- 1. ADIM: Test Verisini Hazırlama ---
print("Test verisi çekiliyor...")
test_verisi = veri_cek_ve_hazirla(
    hisse_kodu=HİSSE_KODU,
    baslangic_tarihi=TEST_BASLANGIC_TARIHI,
    bitis_tarihi=TEST_BITIS_TARIHI
)

if test_verisi is None or len(test_verisi) < 15: # Sequence length'ten büyük olmalı
    print("Test için yeterli veri bulunamadı. Program sonlandırılıyor.")
    exit()

# --- 2. ADIM: Ortamı ve Eğitilmiş Ajanı Kurma ---
print("Ortam ve ajan kuruluyor...")
# Ortamı kurarken sequence_length parametresini de veriyoruz
test_ortami = TicaretOrtami(df=test_verisi, baslangic_bakiye=BASLANGIC_BAKIYE, sequence_length=10)

# --- DÜZELTME 1 ---
# state_size'ı artık bir tuple olarak, yani (10, 5) şeklinde alıyoruz.
state_size = test_ortami.observation_space.shape
action_size = test_ortami.action_space.n

# Ajanı oluşturuyoruz.
agent = DQNAjan(state_size, action_size, model_agirliklari_dosyasi=MODEL_AGIRLIKLARI_DOSYASI)

# --- KRİTİK ADIM: AJANI TEST MODUNA ALMA ---
agent.epsilon = 0.0
print(f"Ajan test moduna alındı (epsilon = {agent.epsilon}).")

# --- 3. ADIM: Test Simülasyonunu Çalıştırma ---
print("Test simülasyonu başlatılıyor...")
state, info = test_ortami.reset()
# --- DÜZELTME 2 ---
# state'i yeniden şekillendirirken 3 boyutlu hale getiriyoruz: (1, 10, 5)
state = np.reshape(state, [1, state_size[0], state_size[1]])

done = False
aksiyon_gecmisi = []

while not done:
    action = agent.act(state)
    aksiyon_gecmisi.append(action)
    
    next_state, reward, terminated, truncated, info = test_ortami.step(action)
    done = terminated or truncated
    
    # next_state'i de yeniden şekillendiriyoruz
    next_state = np.reshape(next_state, [1, state_size[0], state_size[1]])
    state = next_state

print("Test simülasyonu tamamlandı.")

# --- 4. ADIM: Sonuçları Hesaplama ve Karşılaştırma (Bu kısım aynı) ---
dqn_sonuc = info['toplam_portfoy_degeri']

ilk_fiyat = test_verisi['Close'].iloc[0]
son_fiyat = test_verisi['Close'].iloc[-1]
alinabilecek_hisse = BASLANGIC_BAKIYE / ilk_fiyat
al_ve_tut_sonuc = alinabilecek_hisse * son_fiyat

print("\n--- TEST SONUÇLARI ---")
print(f"Başlangıç Bakiyesi: {BASLANGIC_BAKIYE:,.2f} TL")
print("-" * 30)
print(f"DQN Ajanının (LSTM) Portföy Sonucu: {dqn_sonuc:,.2f} TL")
print(f"'Al ve Tut' Stratejisi Sonucu: {al_ve_tut_sonuc:,.2f} TL")
print("-" * 30)

dqn_getiri = ((dqn_sonuc / BASLANGIC_BAKIYE) - 1) * 100
al_ve_tut_getiri = ((al_ve_tut_sonuc / BASLANGIC_BAKIYE) - 1) * 100

print(f"DQN Ajanı (LSTM) Getirisi: %{dqn_getiri:.2f}")
print(f"'Al ve Tut' Getirisi: %{al_ve_tut_getiri:.2f}")

if dqn_getiri > al_ve_tut_getiri:
    print("\nSonuç: DQN ajanı (LSTM), basit 'Al ve Tut' stratejisinden daha iyi bir performans gösterdi. ✅")
else:
    print("\nSonuç: DQN ajanı (LSTM), basit 'Al ve Tut' stratejisini geçemedi. ❌")

# --- 5. ADIM: İşlemleri Görselleştirme (Bu kısım aynı) ---
plt.figure(figsize=(16, 8))
plt.plot(test_verisi.index, test_verisi['Close'], label='THYAO Fiyat', color='blue', alpha=0.6)

# Alım-Satım noktalarını çizerken, simülasyonun başladığı doğru indeksi bulmalıyız.
# Simülasyon, sequence_length'inci günde başlar.
baslangic_indeksi = test_ortami.sequence_length -1

alim_noktalari = [i + baslangic_indeksi for i, a in enumerate(aksiyon_gecmisi) if a == 0]
plt.plot(test_verisi.index[alim_noktalari], test_verisi['Close'].iloc[alim_noktalari], '^', markersize=10, color='green', label='Alım Yapıldı')

satim_noktalari = [i + baslangic_indeksi for i, a in enumerate(aksiyon_gecmisi) if a == 1]
plt.plot(test_verisi.index[satim_noktalari], test_verisi['Close'].iloc[satim_noktalari], 'v', markersize=10, color='red', label='Satım Yapıldı')

plt.title('DQN Ajanının (LSTM) Test Verisi Üzerindeki Alım-Satım Noktaları')
plt.xlabel('Tarih')
plt.ylabel('Hisse Fiyatı (TL)')
plt.legend()
plt.grid(True)
plt.show()