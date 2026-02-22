# Geometry of Reasoning - Development Plan

## 0) Proje Cekirdegi

Bu proje, CoT (Chain-of-Thought) adimlarinda olusan "mantiksal kirilma" anlarini, egitim gerektirmeyen geometri + belirsizlik sinyalleriyle erken tespit etmeyi hedefler.

`Fine-tune` veya klasik `RAG` degil; modelin ic dinamiklerine bakan, matematiksel olarak savunulabilir bir "process-level diagnostics" calismasi olarak konumlanir.

Ana cikti:
- Teknik olarak guclu deney paketi (A/B/C ablation),
- Uctan uca tekrar calistirilabilir kod tabani,
- "Reasoning Seismograph" demo arayuzu,
- Kisa teknik rapor (paper-style).

---

## 1) Temel Terminoloji ve Kavramlar

### Hidden State
Transformer'in her katmanda her token icin olusturdugu temsil vektorudur.
Bu projede analiz edilen ham sinyalin ana kaynagi budur.

### Step Boundary (Adim Siniri)
Token bazli cok gürültülü sinyali azaltmak icin, analiz birimi "adim sonu token" olur.
`Step 1: ...`, `Step 2: ...` gibi zorlanmis formatta her adim icin tek temsil alinacaktir.

### LID (Local Intrinsic Dimension)
Yuksek boyutlu uzayda bir noktanin lokal komsulugunun efektif serbestlik derecesini olcer.
Intuition: LID artisi, temsil geometrisinin daha daginik/kararsiz hale geldigini gosterebilir.

### MLE (Levina-Bickel) - LID Estimator
`k` en yakin komsu mesafeleri ile LID kestirimi:

\[
\hat{m}_k(x)=\left(\frac{1}{k-1}\sum_{i=1}^{k-1}\log\frac{T_k(x)}{T_i(x)}\right)^{-1}
\]

Burada `T_i(x)`, `x` noktasinin `i`-inci en yakin komsu mesafesidir.

### TwoNN - LID/ID Estimator
Sadece ilk iki komsu mesafe oranini kullanir:
- \(\mu_i = r_{i,2} / r_{i,1}\)
- pratik kestirim: \(\hat{d} \approx (\frac{1}{n}\sum_i \log \mu_i)^{-1}\)

Avantaj: minimal hiperparametre hassasiyeti ve kucuk-orneklemde pratik dayaniklilik.

### ABID (Angle-Based Intrinsic Dimension)
Mesafe metriklerinin anlamsizlastigi yuksek boyutta, komsu yon vektorleri arasindaki acisal dagilimi kullanir.
Bu, mesafe tabanli MLE/TwoNN'e ortogonal bir kontrol kanali saglar.

### Effective Dimension (Participation Ratio, PR)
Hidden state kovaryansinin ozdeger spektrumundan global boyut olcumu:

\[
PR = \frac{(\sum_j \lambda_j)^2}{\sum_j \lambda_j^2}
\]

`PR` dususu, temsilin daha az sayida ana yone "cokmesi" anlamina gelebilir.

### Conditional Entropy
Her adimda model logits dagilimindan belirsizlik olcumu:

\[
H_t = - \sum_{v \in V} p(v|x_{\le t}) \log p(v|x_{\le t})
\]

Yuksek entropi tek basina hata demek degildir, ancak LID/PR ile birlikte erken uyari gucunu artirabilir.

---

## 2) Tech Stack ve Araclar

Bu proje icin minimum ama guclu stack:

- Dil/Cekirdek
  - `Python 3.11+`
  - Paket yonetimi: `Poetry`
- Model ve Inference
  - `PyTorch`
  - `transformers`
  - `accelerate`
  - `bitsandbytes` (4-bit/8-bit calisma icin)
- Veri ve Isleme
  - `datasets` (Hugging Face)
  - `numpy`, `scipy`, `pandas`
  - `faiss-cpu` (k-NN hizlandirma; GPU yoksa bile kullanisli)
- Matematiksel Dogrulama
  - `sympy` (step-level equation/result judge)
- Deney ve Istatistik
  - `scikit-learn` (AUROC/AUPRC, calibration, basit modeller)
  - `statsmodels` (istatistiksel testler, gerekirse)
- Gorsellestirme ve Demo
  - `plotly` veya `matplotlib`
  - `gradio` (hizli web demo) veya terminal tabanli `rich`
- Kalite ve Tekrar Edilebilirlik
  - `pytest`
  - `ruff`
  - `mypy` (opsiyonel ama tavsiye)
  - `hydra` veya basit YAML config sistemi

Model secimi (ana deney):
- Reasoning model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Instruct baseline: `Qwen/Qwen2.5-1.5B-Instruct`

Veriseti:
- `openai/gsm8k` (`main` split)
- Opsiyonel ikinci benchmark (MVP sonrasi): `MATH-500` veya benzeri.

---

## 3) Deney Protokolu (Sabit Kurallar)

### Prompt ve Cikti Formatlamasi
- Model zorunlu adim formatiyla uretim yapar: `Step 1:`, `Step 2:`, ...
- Her adimin son token index'i boundary olarak kaydedilir.
- Final cevap tekil bir formatta normalize edilir (judge kolayligi icin).

### Step-level Etiketleme (Ground Truth)
- Her adimdan cikarilan denklem/ara sonuc `sympy` ile dogrulanir.
- Etiket: `1` (dogru adim), `0` (yanlis adim).
- Parse edilemeyen adimlar icin ayrik durum kodu tutulur (`parse_fail`), dogrudan 0'a gokturulmez.

### Hidden State Toplama
- En az 3 katman seviyesi: erken, orta, gec (or. `L/4`, `L/2`, `L`).
- Her adim sinirinda hidden state snapshot alinir.
- Logits ve entropy de ayni anda kaydedilir.

### Reproducibility
- Tum random seed'ler sabitlenir.
- Her kosunun config hash'i ve model revision bilgisi kaydedilir.
- Sonuclar `parquet` formatinda saklanir.

---

## 4) Fazlara Bolunmus Gelistirme Plani 

## Faz 1 - Altyapi ve Deney Iskeleti
Hedef: Repo ve pipeline'in "tek komutla" kosabilir hale gelmesi.

Ciktilar:
- `src/generation/runner.py` ile deterministik generation akisi,
- `src/generation/extraction.py` ile hidden state + logits capture,
- Config sistemi (`configs/*.yaml`),
- Temel test ve lint altyapisi.

Definition of Done:
- Verilen bir GSM8K orneginde adimlar ve hidden state dosyasi olusuyor.
- Ayni config ile tekrar kostugunda ayni metadata uretiliyor.

## Faz 2 - Step-level Judge ve Veri Hazirlama
Hedef: CoT adimlarinin matematiksel dogrulugunu otomatik etiketlemek.

Ciktilar:
- `src/evaluation/sympy_judge.py`
- Adim parser + normalize edici util fonksiyonlar,
- `data/processed/` altinda step-level etiketli tablo.

Definition of Done:
- Rastgele orneklemde manuel kontrol ile judge tutarliligi kabul edilebilir.
- `parse_fail` orani raporlanmis ve dokumante edilmis.

## Faz 3 - Geometri/Belirsizlik Metrik Motoru
Hedef: LID (MLE, TwoNN, ABID), PR, Entropy hesaplayicilarinin dogru ve testli olmasi.

Ciktilar:
- `src/metrics/lid_estimators.py`
- `src/metrics/global_dim.py`
- `src/metrics/uncertainty.py`
- Birim testleri (`tests/test_lid_math.py`, vb.)

Definition of Done:
- Sentetik veri uzerinde beklenen ID trendleri yakalaniyor.
- K degisimlerinde estimator davranisi raporlanabiliyor.

## Faz 4 - Ana Deneyler (A/B/C Ablation)
Hedef: Hipotezin istatistiksel olarak sinanmasi.

Ciktilar:
- Deney A/B/C scriptleri ve sonucu tablolari,
- Erken uyari modeli (hafif logistic/threshold tabanli),
- Grafikler: adim bazli trend, model karsilastirma, `k` hassasiyeti.

Definition of Done:
- Her deney icin birincil metrikler ve confidence interval'lar cikartilmis.
- Sonuclar pozitif ya da negatif olsun, tutarli bir analiz metni uretilmis.

## Faz 5 - Demo, Rapor, Vitrinleme
Hedef: Basvuru dosyasi icin "akilda kalan" arastirma demosu.

Ciktilar:
- `src/visualization/seismograph.py` (canli trend + early warning),
- README'de tek komutla demo calisma akisi,
- Kisa teknik rapor (`report/main.tex`) ve figurlar.

Definition of Done:
- Demo videosu/GIF'i README'de var.
- Harici bir kisi, talimatla reproduksiyon yapabiliyor.

---

## 5) Deneyler ve Metrik Tanimlari

### Deney A - Erken Uyari Sinyali
Soru: Ilk `N` adimdaki sinyaller final dogrulugu tahmin ediyor mu?

Birincil metrikler:
- `AUROC` (final success/failure tahmini),
- `AUPRC` (class imbalance varsa daha bilgilendirici),
- `Lead Time`: ilk alarm adimi ile ilk yanlis adim/final hata arasi fark.

Istatistik:
- Dogru vs yanlis adim dagilimlari icin `Mann-Whitney U`,
- Etki buyuklugu: `Cliff's delta` veya `Cohen's d`.

### Deney B - Reasoning vs Instruct
Soru: Sinyaller reasoning modelde daha iyi kalibre oluyor mu?

Birincil metrikler:
- Ayni pipeline altinda model bazli `AUROC/AUPRC` farki,
- Calibration: `Brier score`, (opsiyonel) `ECE`,
- Signal-to-noise proxy: dogru/yanlis adim ayrisma mesafesi.

### Deney C - Estimator Hassasiyeti
Soru: `k` secimi ve estimator tipi sonuclari ne kadar oynatiyor?

Birincil metrikler:
- `k in {5, 10, 20, 40}` icin performans varyansi,
- Robustness skoru: metriklerin `CV` (coefficient of variation),
- Siralama tutarliligi: estimator ranking icin `Kendall tau`.

---

## 6) Minimum Viable Basari Kriterleri

MVP'in "basarili" sayilmasi icin:
- Uctan uca pipeline tek komutla kosuyor.
- En az bir modelde, erken adim sinyalleri sans seviyesinin anlamli ustunde tahmin gucu veriyor (`AUROC > 0.65` gibi pratik esik).
- Estimator karsilastirmasinda en az bir net trade-off ortaya konuyor (or. TwoNN stabil, MLE daha keskin ama hassas).
- Demo, bir hatali muhakeme akisini erken uyariyla gosterebiliyor.

Not: Negatif sonuc da degerlidir. "Bu sinyal su kosulda calismiyor" bulgusu, dogru deney tasarimiyla arastirma olgunlugu gosterir.

---

## 7) Repo Iskeletini Uygulama Planina Esleme

Onerilen minimal ekler:
- `configs/` (model, generation, metrics, experiment ayarlari)
- `scripts/` (run_generation, run_judge, run_metrics, run_ablation)
- `results/` (tablolar, figurlar, ara ciktilar)

Ana akis:
1. `generation` -> CoT + hidden states toplama
2. `evaluation` -> SymPy ile step-level etiketleme
3. `metrics` -> LID/PR/Entropy cikarimi
4. `experiments` -> A/B/C analizleri
5. `visualization` -> seismograph demo

---

## 8) Teknik Riskler ve Onleyici Kararlar

- Step formatinin bozulmasi
  - Cozum: parser toleransi + strict template + regex fallback.
- SymPy parse hatalari
  - Cozum: normalize katmani, parse_fail etiketi, manuel spot-check.
- Hidden state bellek maliyeti
  - Cozum: secili katmanlar, `float16/bfloat16`, parca parca yazma.
- Modelden modele adim semantigi farki
  - Cozum: ortak prompt protokolu ve model-ozel parse adaptoru.
- P-hacking riski
  - Cozum: metrikleri ve testleri faz basinda sabitle, sonradan degistirme.

---

## 9) Basvuru Dosyasinda Konumlama (CV/GitHub)

Bu proje su 3 mesaji net verir:
- Matematiksel derinlik: ID tahmini, topolojik sinyal, istatistiksel test.
- Arastirma disiplini: ablation, negatif sonuca acik metodoloji, reproduksiyon.
- Urunlestirme: offline judge + canli seismograph demo + temiz repo mimarisi.

README'de tek satirlik konumlama onerisi:
"Training-free geometric diagnostics for early detection of reasoning derailment in LLM chain-of-thought trajectories."
