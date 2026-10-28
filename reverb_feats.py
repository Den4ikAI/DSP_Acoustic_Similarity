"""Временные признаки реверберации: скорость затухания, а не форма спектра.

ПОЧЕМУ ОТДЕЛЬНО ОТ ТЕКСТУРЫ. Прежние «текстурные» признаки были временными лишь
формально: модуляционный спектр по 8 широким полосам в 6 логарифмических бинах —
слишком грубо, чтобы увидеть хвост. Замер это подтвердил: комната у них 0.638
против 0.620 у сырого LTAS, то есть почти ничего, при том что DRR они улучшали
заметно. DRR — отношение энергий, он живёт в спектре; T60 — скорость затухания,
и её в усреднённых по времени статистиках нет физически.

ЧТО СЧИТАЕМ. Всё — на огибающих в дБ по четырём широким полосам, 100 кадров/с.

  скорость спада — распределение приращений уровня на разных промежутках
      (10, 30, 100 мс). Ключ здесь в НИЖНИХ процентилях: в гулкой комнате звук
      физически не может упасть быстрее, чем позволяет хвост, поэтому самый
      крутой наблюдаемый спад — это прямая слепая оценка T60. Верхние процентили
      дают атаки: реверберация их не трогает, и разница атака/спад сама по себе
      информативна;
  автокорреляция огибающей на задержках 20..400 мс — реверберация размазывает
      огибающую и поднимает корреляцию на коротких задержках;
  отношение модуляционных полос 4–20 Гц к 20–50 Гц — слепая мера разборчивости
      в духе SRMR: хвост съедает быстрые модуляции первыми.

Ничего из этого не выводится из среднего спектра, и это главное.
"""
import torch
import torchaudio

SR = 16000
HOP = 160                      # 100 кадров в секунду
GROUPS = 4                     # 64 мела -> 4 широкие полосы
GAPS = [1, 3, 10]              # 10, 30, 100 мс
PCTS = [0.02, 0.10, 0.50, 0.90, 0.98]
LAGS = [2, 5, 10, 20, 40]      # 20, 50, 100, 200, 400 мс


class ReverbFeats:
    def __init__(self, device="cuda:0", n_mels=64):
        self.dev = device
        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=512, hop_length=HOP, n_mels=n_mels).to(device)
        self.dim = GROUPS * (len(GAPS) * len(PCTS) + len(LAGS) + 2)

    def __call__(self, x):
        p = self.mel(x)
        T = p.shape[1]
        g = p.reshape(GROUPS, self.n_mels // GROUPS, T).mean(1)
        e = 10 * torch.log10(g + 1e-10)                    # огибающие в дБ
        q = torch.tensor(PCTS, device=x.device)
        out = []

        # 1) распределение приращений уровня на разных промежутках
        for gap in GAPS:
            if T <= gap + 1:
                out.append(torch.zeros(GROUPS * len(PCTS), device=x.device))
                continue
            d = (e[:, gap:] - e[:, :-gap]) * (100.0 / gap)      # дБ в секунду
            out.append(torch.quantile(d, q, dim=1).T.reshape(-1))

        # 2) автокорреляция огибающей: реверберация размазывает её во времени
        c = e - e.mean(1, keepdim=True)
        nrm = (c.pow(2).mean(1) + 1e-8).sqrt()
        ac = []
        for L in LAGS:
            if T <= L + 1:
                ac.append(torch.zeros(GROUPS, device=x.device))
                continue
            ac.append((c[:, L:] * c[:, :-L]).mean(1) / (nrm * nrm + 1e-8))
        out.append(torch.stack(ac, 1).reshape(-1))

        # 3) отношение модуляционных полос: хвост съедает быстрые модуляции
        M = torch.fft.rfft(c * torch.hann_window(T, device=x.device), dim=1).abs() ** 2
        fr = torch.arange(M.shape[1], device=x.device) * (100.0 / T)
        lo = (fr >= 4) & (fr < 20)
        hi = (fr >= 20) & (fr < 50)
        r = torch.log((M[:, lo].mean(1) + 1e-8) / (M[:, hi].mean(1) + 1e-8))
        # и асимметрия атака/спад — реверберация трогает только спад
        d1 = e[:, 1:] - e[:, :-1]
        up = torch.where(d1 > 0, d1, torch.zeros_like(d1)).mean(1)
        dn = torch.where(d1 < 0, -d1, torch.zeros_like(d1)).mean(1)
        out.append(torch.stack([r, torch.log((up + 1e-6) / (dn + 1e-6))], 1).reshape(-1))
        return torch.cat(out)


class DecayCurve:
    """Средняя кривая затухания после спадов речи — слепая огибающая комнаты.

    Идея прямая: найти моменты, где речь оборвалась (уровень был высоким и пошёл
    вниз), и усреднить то, что происходит с энергией в следующие 400 мс. В сухой
    записи она падает почти отвесно до шумовой полки; в гулкой тянется хвост. Это
    не описание реверберации через спектр, а её измерение по определению.

    Кривая берётся в дБ относительно момента спада и нормируется на шумовую полку
    полосы, чтобы уровень записи и её громкость не подмешивались в форму.
    """

    def __init__(self, device="cuda:0", n_mels=64, span=40, npt=8):
        self.dev = device
        self.n_mels = n_mels
        self.span = span                       # 400 мс после спада
        self.npt = npt
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=512, hop_length=HOP, n_mels=n_mels).to(device)
        self.dim = GROUPS * npt

    def __call__(self, x):
        p = self.mel(x)
        T = p.shape[1]
        g = p.reshape(GROUPS, self.n_mels // GROUPS, T).mean(1)
        e = 10 * torch.log10(g + 1e-10)
        floor = torch.quantile(e, 0.1, dim=1, keepdim=True)
        hi = torch.quantile(e, 0.75, dim=1, keepdim=True)
        out = []
        idx = torch.arange(self.span, device=x.device)
        pick = (idx * (self.span - 1) / max(self.npt - 1, 1)).long().clamp(max=self.span - 1)
        pick = torch.linspace(0, self.span - 1, self.npt, device=x.device).long()
        for b in range(GROUPS):
            eb = e[b]
            # момент спада: уровень выше верхней четверти, а через 50 мс упал
            if T <= self.span + 6:
                out.append(torch.zeros(self.npt, device=x.device))
                continue
            cand = ((eb[:-self.span] > hi[b, 0]) &
                    (eb[5:5 - self.span] < eb[:-self.span] - 3)).nonzero().flatten()
            if len(cand) < 3:
                out.append(torch.zeros(self.npt, device=x.device))
                continue
            seg = torch.stack([eb[int(c):int(c) + self.span] for c in cand[:200]])
            seg = seg - seg[:, :1]                       # относительно момента спада
            rng = (hi[b, 0] - floor[b, 0]).clamp(min=1.0)
            out.append((seg.mean(0) / rng)[pick])
        return torch.cat(out)
