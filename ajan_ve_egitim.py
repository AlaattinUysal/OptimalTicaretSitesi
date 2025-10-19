import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM # LSTM katmanını import ettik
from keras.optimizers import Adam
import random
from collections import deque

# Diğer dosyalarımızdan gerekli sınıfları ve fonksiyonları import ediyoruz
from veri_hazirlama import veri_cek_ve_hazirla
from ticaret_ortami import TicaretOrtami

# Modelin tekrarlanabilir sonuçlar üretmesi için random seed'leri ayarlıyoruz
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DQNAjan:
    """
    LSTM tabanlı, hafıza yeteneğine sahip Derin Q-Network Ajanı.
    """
    def __init__(self, state_size, action_size, model_agirliklari_dosyasi="dqn_model.weights.h5"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model_agirliklari_dosyasi = model_agirliklari_dosyasi
        
        # Hyperparametreler
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.97 # Daha hızlı öğrenme için güncellenmişti
        
        # Modeli oluştur
        self.model = self._build_model()

        # Eğer kayıtlı ağırlık dosyası varsa, yükle ve eğitime devam et.
        # Yoksa, eğitime sıfırdan başla.
        if os.path.exists(self.model_agirliklari_dosyasi):
            print(f"'{self.model_agirliklari_dosyasi}' bulundu. Ağırlıklar yükleniyor...")
            self.model.load_weights(self.model_agirliklari_dosyasi)
            self.epsilon = 0.2
        else:
            print("Mevcut ağırlık dosyası bulunamadı. Eğitime sıfırdan başlanıyor.")
            self.epsilon = 1.0

    def _build_model(self):
        """ LSTM tabanlı yeni sinir ağı modeli """
        model = Sequential()
        
        # Girdi olarak (10, 5) şeklinde bir matris alacak olan LSTM katmanı
        # input_shape'i ortamdan gelen verinin şekline göre ayarlıyoruz.
        # self.state_size burada (10, 5) gibi bir tuple olacak.
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.state_size[0], self.state_size[1])))
        model.add(LSTM(units=50))
        
        # LSTM'den gelen özeti işleyecek olan standart katmanlar
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        print("LSTM tabanlı yeni model başarıyla oluşturuldu.")
        model.summary() # Modelin mimarisini terminale yazdır
        return model

    def remember(self, state, action, reward, next_state, done):
        """ Ajanın tecrübelerini hafızaya kaydeder """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """ Epsilon-Greedy stratejisi ile aksiyon seçimi """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """ Hafızadan rastgele bir alt küme (batch) ile modeli eğitir """
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # state'ler ve next_state'ler artık (batch_size, sequence_length, features) şeklinde olacak
        # Bu yüzden eğitimi daha verimli hale getirmek için toplu (batch) eğitim yapabiliriz.
        # Şimdilik basitlik adına tek tek eğitime devam edelim.
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- ANA EĞİTİM DÖNGÜSÜ ---
if __name__ == "__main__":
    # Parametreler
    EPISODES = 50 
    BATCH_SIZE = 32
    
    # Veriyi hazırla
    hisse_verisi = veri_cek_ve_hazirla(
        hisse_kodu="THYAO.IS",
        baslangic_tarihi="2020-01-01",
        bitis_tarihi="2024-12-31"
    )
    
    if hisse_verisi is not None:
        # Ortamı ve Ajanı oluştur
        env = TicaretOrtami(df=hisse_verisi, baslangic_bakiye=100000)
        # state_size artık bir tuple (örn: (10, 5)) olacak
        state_size = (env.observation_space.shape[0], env.observation_space.shape[1])
        action_size = env.action_space.n
        agent = DQNAjan(state_size, action_size)
        
        # Eğitim Döngüsünü Başlat
        try:
            for e in range(EPISODES):
                state, info = env.reset()
                state = np.reshape(state, [1, state_size[0], state_size[1]])
                
                done = False
                while not done:
                    action = agent.act(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    next_state = np.reshape(next_state, [1, state_size[0], state_size[1]])
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    if done:
                        final_portfolio = info['toplam_portfoy_degeri']
                        print(f"Bölüm: {e+1}/{EPISODES}, Portföy Değeri: {final_portfolio:.2f}, Epsilon: {agent.epsilon:.2f}")
                
                agent.replay(BATCH_SIZE)
                
                if (e + 1) % 10 == 0:
                    print(f"Checkpoint: {e+1}. bölüm tamamlandı. Model ağırlıkları kaydediliyor...")
                    agent.model.save_weights(agent.model_agirliklari_dosyasi)
                    
        finally:
            print("Eğitim tamamlandı veya durduruldu. Son model ağırlıkları kaydediliyor.")
            agent.model.save_weights(agent.model_agirliklari_dosyasi)
    else:
        print("Veri çekilemediği için program sonlandırılıyor.")