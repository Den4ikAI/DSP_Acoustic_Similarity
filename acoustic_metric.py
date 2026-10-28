"""Метрика акустического сходства: тракт записи отдельно от голоса.

ЗАЧЕМ ОТДЕЛЬНАЯ МЕТРИКА. Speaker similarity на готовых эмбеддерах наполовину
меряет запись, а не голос: на бенчмарке реальных трактов CAM++ в 64 % случаев
кладёт двух РАЗНЫХ людей, записанных одним трактом, ближе, чем пару с разными
трактами (ReDimNet2-B6 — 58 %, он же после дообучения — 66 %). Поэтому модель,
которая лучше скопировала акустику донора, выигрывает по speaker similarity, даже
если тембр у неё хуже. Эта метрика меряет как раз ту ось и в паре с ReDimNet2-B6
почти зеркальна ему: 0.911 против 0.566 по тракту и диктору соответственно.

ДВА ЧИСЛА, А НЕ ОДНО. Тракт распадается на две части, которые ведут себя
по-разному и не заменяют друг друга (корреляция их косинусов 0.53, на трети пар
они расходятся):

  СПЕКТРАЛЬНАЯ (133) — 133 статистики лог-мела: среднее и разброс по времени по
      64 полосам, квантили, наклон. Опознаёт подложку, эквализацию, полосу,
      уровень шума. На кино это единственное, что работает (AUC 0.777), потому
      что дубляж пишут в сухих кабинах и тракт там — это сведение;
  ВРЕМЕННА́Я (88) — распределение скорости спада уровня на промежутках 10/30/100 мс,
      автокорреляция огибающей на задержках 20..400 мс, отношение модуляционных
      полос 4–20 к 20–50 Гц, асимметрия атака/спад. Опознаёт ПОМЕЩЕНИЕ. На
      реальных импульсных характеристиках 88 временных измерений в одиночку бьют
      все 133 спектральных (0.713 против 0.659) и текут диктором заметно меньше
      (0.553 против 0.604).

Реверберация — это скорость затухания, и в усреднённых по времени статистиках её
нет физически. Отсюда и разделение: спрашивать «похожа ли акустика» одним числом
значит смешивать комнату с микшером.

    m = AcousticMetric()
    a, b = m.encode("ref.wav"), m.encode("synth.wav")
    spec, rev = m.similarity_parts(a, b)     # похоже сведение / похоже помещение
    print(m.similarity(a, b))                # среднее из двух

РАБОЧИЙ ДИАПАЗОН спектральной оси на кино: у двух клипов одного фильма медиана
косинуса +0.63 (квартили +0.39..+0.79), у клипов разных фильмов — 0.00
(-0.33..+0.33). Шкала читается напрямую, без калибровки.

Обучаемый код здесь намеренно НЕ используется. Восемь постановок обучения (от
дистилляции до свёртки по спектрограмме) сырые статистики не превзошли: на
синтетических трактах обученный код доходит до 0.986, но на реальном кино падает
до 0.718 при 0.777 у статистик. Подробности — в RESULTS.md.

Длина записи: с 3 секунд цифры стабильны, короче 2 секунд метрике верить не стоит.
"""
import os
import sys

import numpy as np
import torch
import torchaudio

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reverb_feats import ReverbFeats            # noqa: E402


class AcousticMetric:
    def __init__(self, device: str = "cuda:0", norm: str = None):
        self.dev = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=512, hop_length=160, n_mels=64).to(device)
        self.rev = ReverbFeats(device)
        z = np.load(norm or os.path.join(HERE, "acoustic_norm.npz"))
        self.dim_spec = int(z["dim_spec"])
        self.dim_rev = int(z["dim_rev"])
        self.mu = torch.tensor(z["mu"], device=device)
        self.sd = torch.tensor(z["sd"], device=device)

    def _wave(self, src):
        if isinstance(src, str):
            w, sr = torchaudio.load(src)
            return torchaudio.functional.resample(
                w.mean(0, keepdim=True), sr, SR)[0].to(self.dev)
        x = torch.as_tensor(src, dtype=torch.float32, device=self.dev)
        return x.mean(0) if x.dim() > 1 else x

    def encode(self, src) -> np.ndarray:
        x = self._wave(src)
        # уровень уравнивается по RMS: пиковая нормировка утекает текстурой —
        # импульсный шум сажает громкость всей записи
        x = x / x.pow(2).mean().clamp(min=1e-12).sqrt() * 0.05
        log = torch.log(self.mel(x) + 1e-8)
        fe = log.mean(0)
        q = torch.quantile(fe, torch.tensor([0.1, 0.5, 0.9], device=self.dev))
        d = fe[1:] - fe[:-1]
        neg = (d[d < 0].mean().reshape(1) if bool((d < 0).any())
               else torch.zeros(1, device=self.dev))
        spec = torch.cat([log.mean(1), log.std(1), q, neg, d.std().reshape(1)])
        f = torch.cat([spec, self.rev(x)])
        return ((f - self.mu) / self.sd).cpu().numpy()

    def encode_batch(self, srcs) -> np.ndarray:
        return np.stack([self.encode(s) for s in srcs])

    @staticmethod
    def _cos(a, b):
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))

    def similarity_parts(self, a: np.ndarray, b: np.ndarray):
        """(сведение, помещение) — две оси тракта по отдельности."""
        k = self.dim_spec
        return self._cos(a[:k], b[:k]), self._cos(a[k:], b[k:])

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        s, r = self.similarity_parts(a, b)
        return 0.5 * (s + r)
