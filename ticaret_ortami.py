import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TicaretOrtami(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, baslangic_bakiye=10000, islem_maliyeti=0.001, **kwargs):
        super(TicaretOrtami, self).__init__()
        self.df = df
        self.baslangic_bakiye = baslangic_bakiye
        self.islem_maliyeti = islem_maliyeti
        self.action_space = spaces.Discrete(3)

        # Gözlem alanı 6 özellikli tek boyutlu vektör
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # Scaler'ı duygu skoru dahil tüm piyasa verileriyle eğitiyoruz
        self.scaler.fit(self.df[['Close', 'SMA_14', 'RSI_14', 'sentiment_score']])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.bakiye = self.baslangic_bakiye
        self.hisse_sayisi = 0
        self.mevcut_adim = 0
        self.toplam_portfoy_degeri = self.baslangic_bakiye
        obs = self._sonraki_gozlem()
        info = self._get_info()
        return obs, info

    def _sonraki_gozlem(self):
        piyasa_verisi_raw = self.df[['Close', 'SMA_14', 'RSI_14', 'sentiment_score']].iloc[self.mevcut_adim:self.mevcut_adim+1]
        piyasa_verisi_scaled = self.scaler.transform(piyasa_verisi_raw)[0]
        
        bakiye_scaled = (self.bakiye / self.baslangic_bakiye) * 2 - 1 # Bakiyeyi de -1 ile 1 arasına ölçekleyelim
        hisse_sayisi_scaled = self.hisse_sayisi / 1000 # Hisse sayısını da ölçekleyelim
        
        obs = np.concatenate([piyasa_verisi_scaled, [bakiye_scaled, hisse_sayisi_scaled]]).astype(np.float32)
        return obs

    def _get_info(self):
        return {'toplam_portfoy_degeri': self.toplam_portfoy_degeri}

    def step(self, action):
        onceki_portfoy_degeri = self.toplam_portfoy_degeri
        if self.mevcut_adim >= len(self.df) - 1:
            return self._sonraki_gozlem(), 0, True, False, self._get_info()
        
        mevcut_fiyat = self.df['Close'].iloc[self.mevcut_adim]
        if action == 0: # Al
            if self.bakiye >= mevcut_fiyat * (1 + self.islem_maliyeti):
                self.bakiye -= mevcut_fiyat * (1 + self.islem_maliyeti)
                self.hisse_sayisi += 1
        elif action == 1: # Sat
            if self.hisse_sayisi > 0:
                self.bakiye += mevcut_fiyat * (1 - self.islem_maliyeti)
                self.hisse_sayisi -= 1
        
        self.toplam_portfoy_degeri = self.bakiye + (self.hisse_sayisi * mevcut_fiyat)
        self.mevcut_adim += 1
        
        # Basit ödül fonksiyonuna geri dönüyoruz
        reward = self.toplam_portfoy_degeri - onceki_portfoy_degeri
        
        terminated = self.mevcut_adim >= len(self.df) - 1
        truncated = False
        obs = self._sonraki_gozlem()
        info = self._get_info()
        return obs, reward, terminated, truncated, info