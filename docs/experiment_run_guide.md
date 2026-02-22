Faz 4’ü gerçekten “raporlanabilir” hale getirmek için en doğru yol, bunu **3 aşamada** koşmak:

1) küçük pilot (pipeline sağlam mı),  
2) orta ölçek (metrikler anlamlı mı),  
3) final ölçek (rapor/CV çıktısı).

Aşağıya birebir uygulayabileceğin bir çalışma rehberi bırakıyorum.

---

## 1) Hedefin Net Tanımı (Faz 4)

`run_ablation.py` ile 3 deneyin de çıktısını üretmek:

- **A (Early Warning):** İlk `N` adım sinyalinden final hatayı tahmin (`AUROC/AUPRC/Brier/ECE`, lead-time)
- **B (Model Comparison):** Model bazlı kıyas
- **C (k Sensitivity):** `k` değişince performans stabil mi (`auroc_cv_over_k`, Kendall tau)

---

## 2) Ön Kontrol

Terminalde proje kökünde:

```bash
poetry install
poetry run pytest
```

Apple Silicon için (MPS limitlerinden kaçınmak adına) çalıştırırken `GOR_DEVICE=cpu` kullan.

---

## 3) Aşamalarla Çalıştırma

### Aşama A — Hızlı Pilot (tek model, çok küçük veri)

```bash
GOR_DEVICE=cpu poetry run python scripts/run_ablation.py \
  --experiment ALL \
  --models Qwen/Qwen2.5-0.5B-Instruct \
  --split test \
  --start-index 0 \
  --num-samples 8 \
  --max-new-tokens 192 \
  --k-values 5,10,20,40 \
  --primary-k 10 \
  --early-n 2 \
  --seed 42 \
  --out results/ablation_pilot
```

Beklenen: dosyalar oluşmalı (`ablation_summary.json`, `experiment_a_*`, `experiment_b_*`, `experiment_c_*`).

---

### Aşama B — Orta Ölçek (iki model, anlamlı sinyal var mı)

Önce tek komutta iki model deneyebilirsin:

```bash
GOR_DEVICE=cpu poetry run python scripts/run_ablation.py \
  --experiment ALL \
  --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,Qwen/Qwen2.5-0.5B-Instruct \
  --split test \
  --start-index 0 \
  --num-samples 40 \
  --max-new-tokens 192 \
  --k-values 5,10,20,40 \
  --primary-k 10 \
  --early-n 2 \
  --seed 42 \
  --out results/ablation_mid
```

Eğer bu çok uzun sürerse modeli ayır:
- bir koşu `DeepSeek...`
- bir koşu `Qwen...`
- sonra step tablolarını birleştirip `--input` ile tekrar analiz çalıştır.

---

### Aşama C — Final Koşu (rapor için)

```bash
GOR_DEVICE=cpu poetry run python scripts/run_ablation.py \
  --experiment ALL \
  --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,Qwen/Qwen2.5-0.5B-Instruct \
  --split test \
  --start-index 40 \
  --num-samples 80 \
  --max-new-tokens 192 \
  --k-values 5,10,20,40 \
  --primary-k 10 \
  --early-n 2 \
  --seed 42 \
  --out results/ablation_final
```

---

## 4) Sonuçları Nasıl Yorumlayacaksın?

### Experiment A (`experiment_a_summary.json`)
- `logistic.auroc`, `logistic.auprc` yüksekse iyi.
- `baseline_threshold` ile kıyasla: logistic belirgin daha iyi olmalı.
- `lead_time` pozitifse “erken uyarı” iddian güçlenir.

### Experiment B (`experiment_b_model_comparison.csv`)
- Reasoning modelin (`DeepSeek`) `auroc/auprc` avantajı var mı?
- `brier` ve `ece` daha düşükse kalibrasyon daha iyi.
- `signal_gap_mean` ve `cohen_d` ayrışma kalitesini gösterir.

### Experiment C (`experiment_c_k_sensitivity.csv/.json`)
- `auroc_cv_over_k` düşükse (stabilite iyi).
- Kendall tau yüksekse (`~0.6+` iyi bir işaret) sıralama tutarlı.

---

## 5) Sık Sorunlar ve Çözüm

- **`AUROC`/`AUPRC` = `NaN`**  
  Genelde tek sınıf çıkmıştır (hepsi doğru/yanlış).  
  Çözüm: `num-samples` artır, farklı `start-index` dene.

- **Koşu çok yavaş**  
  `max-new-tokens` 192 -> 128 yap; modeli ayrı ayrı çalıştır.

- **Step sinyali az çıkıyor**  
  Çözüm: `max-new-tokens` artır; step formatı bozuluyorsa prompt/generation ayarını sıkı tut.

- **MPS/Memory hatası**  
  `GOR_DEVICE=cpu` ile devam et (şu an en stabil seçenek).

---

## 6) Başvuru Dosyası İçin “Minimum Güçlü Paket”

Finalde şunları rapora koy:
- A/B/C için tek toplu tablo (ana metrikler)
- 1-2 grafik (`experiment_b_model_comparison.html`, `experiment_c_*`)
- Negatif bulguları da yaz (özellikle `k` hassasiyeti ve parse-fail etkisi)

---

İstersen bir sonraki adımda sana **“final koşu komutlarını makinene göre optimize edilmiş”** tek bir runbook (kopyala-yapıştır) çıkarayım: kısa/orta/uzun profil şeklinde.