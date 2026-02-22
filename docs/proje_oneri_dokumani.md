# 📄 PROJE ÖNERİSİ: Muhakemenin Geometrisi 
**Alt Başlık:** *Zincirleme Düşünce (CoT) Sürecinde Mantıksal Kırılmaların Erken Uyarı Sinyali Olarak İçsel Boyut ve Entropi Dinamikleri*

## 1. Merkez Hipotez (Thesis Statement)
Modern *Reasoning LLM*’lerin (örn. DeepSeek-R1) problem çözerken ürettikleri her bir mantıksal adımın doğruluğunu ve "raydan çıkma" (derailment/hallucination) anlarını; modelin gizli durumlarındaki (hidden states) eğitim gerektirmeyen topolojik sinyaller (Yerel İçsel Boyut, Efektif Boyut ve Entropi) ile yüksek doğrulukla tahmin edebiliriz.

## 2. Matematiksel Çerçeve ve Metrik Paketi
Yüksek boyutlu uzaylarda (curse of dimensionality) tek bir metriğe bağlı kalma riskini elimine etmek için, token/adım bazında aşağıdaki 3'lü "Geometri ve Belirsizlik" paketi hesaplanacaktır:

*   **(A) LID (Local Intrinsic Dimension) Estimatorleri:** 
    *   *Levina-Bickel MLE:* Mesafe tabanlı klasik maksimum olabilirlik kestirimi.
    *   *TwoNN:* Eğriliğe ve veri yoğunluğu (density) değişimlerine dayanıklı minimal komşuluk yaklaşımı.
    *   *ABID (Angle-Based ID):* Mesafelerin anlamını yitirdiği yüksek boyutlarda, vektörler arası açısal dağılımı kullanan ortogonal bir kontrol metriği.
*   **(B) Efektif Boyut (Effective Dimension):** 
    *   *Participation Ratio (PR):* Gizli durumların kovaryans matrisinin özdeğer (eigenvalue) spektrumu üzerinden global yörünge boyutunun ölçümü.
*   **(C) Belirsizlik (Uncertainty):** 
    *   *Conditional Entropy:* Modelin logit dağılımı üzerinden anlık bilgi entropisi.

## 3. Operasyonel Tanımlar ve Değerlendirme (Evaluation)
*   **Adım Sınırları (Step Boundary):** Gürültüyü azaltmak için analizler token bazında değil, prompt mühendisliği ile zorlanmış adım (`Step 1:`, `Step 2:`) sınırlarında (her adımın son token'ında) agrege edilecektir.
*   **Doğruluk Etiketi (Step-level Correctness):** Modelin ürettiği matematiksel denklemler/çıktılar, *SymPy* kütüphanesi ile offline olarak doğrulanarak her adıma (0: Yanlış, 1: Doğru) şeklinde kesin etiketler atanacaktır (GSM8K veri seti üzerinde).

## 4. Deney Tasarımı (Ablation Çalışmaları)
Proje, tek bir doğrusal iddia yerine şu üç kontrollü deneyi (ablation) içerir:
*   **Deney A (Görev Zorluğu ve Erken Uyarı):** Doğru giden adımlar ile yanlışa sapan adımlar arasındaki LID trendlerinin istatistiksel ayrışması. Sinyaller ilk $N$ adımda final başarısını (erken uyarı) tahmin edebilir mi?
*   **Deney B (Reasoning vs. Instruct):** Sinyallerin kalibrasyonu; CoT yeteneği olan *DeepSeek-R1-Distill-1.5B* ile standart *Qwen-1.5B-Instruct* arasında karşılaştırılacaktır.
*   **Deney C (Estimator Hassasiyeti):** LID hesaplamalarında $k$ (komşu sayısı) hiperparametresinin MLE, TwoNN ve ABID üzerindeki tutarlılık (robustness) analizi.

## 5. Proje Çıktısı ve Demo: "Reasoning Seismograph"
GitHub reposunun vitrini olarak, modelin adım adım çıktı ürettiği bir arayüzde (Terminal veya basit UI); eş zamanlı olarak LID, PR ve Entropi değerlerini çizen ve SymPy kontrolüyle mantıksal kırılma anında kırmızı "Erken Uyarı (Early Warning)" sinyali veren bir sismograf animasyonu sunulacaktır.

---

## 🛠️ GitHub Repository İskeleti (ETH/EPFL Standartlarında)

Bu araştırma, akademik yazılım mühendisliği prensipleriyle aşağıdaki modüler yapıda inşa edilecektir:

```text
geometry-of-reasoning/
├── data/
│   ├── raw/                  # GSM8K veri setinin işlenmemiş alt kümesi
│   └── processed/            # SymPy ile etiketlenmiş step-level veriler
├── notebooks/
│   ├── 01_baseline_evaluation.ipynb # SymPy etiketleme ve veri keşfi (LaTeX notasyonlu)
│   └── 02_metric_distributions.ipynb # LID vs Entropi dağılım analizleri
├── report/
│   ├── main.tex              # NeurIPS 2025/2026 formatında teknik makale
│   └── references.bib        # ProcessLID, TwoNN, ABID referansları
├── src/
│   ├── __init__.py
│   ├── generation/
│   │   ├── runner.py         # HuggingFace üzerinden <think> tag'ini zorlayan wrapper
│   │   └── extraction.py     # İlgili katmanlardan hidden state'leri toplayan hook'lar
│   ├── metrics/
│   │   ├── lid_estimators.py # MLE, TwoNN ve ABID matematiksel implementasyonları
│   │   ├── global_dim.py     # Participation Ratio (PR) hesaplayıcısı
│   │   └── uncertainty.py    # Logit tabanlı entropi hesaplayıcısı
│   ├── evaluation/
│   │   └── sympy_judge.py    # Step-level matematiksel doğruluğu ölçen offline judge
│   └── visualization/
│   │   └── seismograph.py    # Sismograf animasyonu / grafik oluşturucu
├── tests/
│   ├── test_lid_math.py      # Kendi yazdığınız ABID/TwoNN'in birim testleri (pytest)
│   └── test_sympy_judge.py   # Değerlendirme modülünün testleri
├── pyproject.toml            # Modern Python bağımlılık yönetimi (Poetry/uv için)
└── README.md                 # Sismograf GIF'i, matematiksel özet ve repoyu çalıştırma rehberi
```

---

### Neden Bu Tasarım "Kabul" Getirir?
1. **Araştırma Olgunluğu:** "Model çökecek" demek yerine "Hangi estimator nerede çalışıyor?" diyerek negatif sonuç riskini sıfırladınız.
2. **Matematik Şovu:** `src/metrics/` altındaki kendi yazdığınız 4 farklı boyut hesaplama algoritması, Matematik Mühendisliği diplomanızın kodlanmış halidir.
3. **Mühendislik Pratiği:** "Offline Judge" (SymPy) ve modüler test edilmiş mimari, sadece jupyter'de kod çalıştıran biri olmadığınızı, uçtan uca sistem kurabildiğinizi kanıtlar.