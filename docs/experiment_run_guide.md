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

Opsiyonel: bitsandbytes ile quantization deneyeceksen (CUDA ortamlarında):

```bash
poetry install -E quantization
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
  --primary-k 20 \
  --analysis-layer late \
  --bootstrap-iters 400 \
  --bootstrap-alpha 0.05 \
  --early-n 2 \
  --verbose \
  --log-every 1 \
  --checkpoint-every 1 \
  --torch-threads 10 \
  --torch-interop-threads 2 \
  --seed 42 \
  --out results/ablation_pilot
```

Beklenen: dosyalar oluşmalı (`ablation_summary.json`, `experiment_a_*`, `experiment_b_*`, `experiment_c_*`).

Yeni Experiment A çıktıları içinde özellikle şunlara bak:
- `experiment_a_alarm_policy_comparison.csv`
- `experiment_a_threshold_sweep.csv`
- `experiment_a_raw_vs_calibrated.csv`
- `experiment_a_reliability_curve.csv`

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
  --primary-k 20 \
  --analysis-layer late \
  --bootstrap-iters 400 \
  --bootstrap-alpha 0.05 \
  --early-n 2 \
  --verbose \
  --log-every 2 \
  --checkpoint-every 1 \
  --torch-threads 10 \
  --torch-interop-threads 2 \
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
  --primary-k 20 \
  --analysis-layer late \
  --bootstrap-iters 400 \
  --bootstrap-alpha 0.05 \
  --early-n 2 \
  --verbose \
  --log-every 5 \
  --checkpoint-every 1 \
  --torch-threads 10 \
  --torch-interop-threads 2 \
  --seed 42 \
  --out results/ablation_final
```

Checkpoint/Resume notu:
- Script, her model için adım satırlarını koşu sırasında anlık olarak kaydeder.
- Varsayılan davranış resume’dur; aynı komutu tekrar verdiğinde işlenmemiş indekslerden devam eder.
- Temiz başlangıç için `--no-resume` kullan.
- Model bazlı çıktılar `results/<run>/models/<model-safe>/` altında tutulur.
- Yeni şemada step tablosu `step_text`, `normalized_step_text` ve `matched_values` alanlarını da taşır; parser failure bank ve demo scripti bunları kullanır.
- Step tabloları artık hem `csv` hem `parquet` olarak yazılır.
- Root seviyesinde `run_metadata.json`, model klasörlerinde `model_metadata.json` üretilir.
- `experiment_a_layer_comparison.*` ve `experiment_b_model_comparison_best_layer.*` dosyaları katman-odaklı okumayı kolaylaştırır.

---

## 4) Sonuçları Nasıl Yorumlayacaksın?

### Experiment A (`experiment_a_summary.json`)
- `logistic.auroc`, `logistic.auprc` yüksekse iyi.
- `logistic_ci.*` aralıkları darsa tahmin daha kararlı.
- `baseline_threshold` ile kıyasla: logistic belirgin daha iyi olmalı.
- `lead_time` pozitifse “erken uyarı” iddian güçlenir.
- `selected_alarm_policy` alanı, step-level alarm kararında seçilen policy’yi gösterir.
- `lead_time` altında artık `false_alarm_before_any_error_rate`, `late_alarm_rate`, `missed_alarm_rate`, `first_alarm_step_mean`, `first_error_step_mean` gibi timing odaklı alanlar da var.
- `calibration.rows` içinde `raw`, `platt`, `isotonic` karşılaştırmasını görürsün.

### Experiment B (`experiment_b_model_comparison_common_index.csv`)
- Birden fazla model varsa ana kıyas için common-index tablosunu kullan.
- `experiment_b_model_comparison.csv` dosyası varsayılan olarak common-index’i (varsa) işaret eder.
- `experiment_b_model_comparison_best_layer.csv` ise her modeli kendi en iyi doğrulanan katmanında gösterir; sabit-layer kıyasın yerine geçmez, onu tamamlar.
- Reasoning modelin (`DeepSeek`) `auroc/auprc` avantajı var mı?
- `brier` ve `ece` daha düşükse kalibrasyon daha iyi.
- `signal_gap_mean` ve `cohen_d` ayrışma kalitesini gösterir.

### Experiment C (`experiment_c_k_sensitivity.csv/.json`)
- `auroc_cv_over_k` düşükse (stabilite iyi).
- Kendall tau yüksekse (`~0.6+` iyi bir işaret) sıralama tutarlı.

---

## 5) Offline Yeniden Analiz

Elinde step tablosu varsa modeli tekrar çalıştırmadan yalnızca analiz katmanını koşabilirsin:

```bash
poetry run python scripts/run_ablation.py \
  --experiment A \
  --input results/ablation_final/step_signal_table.parquet \
  --out results/ablation_reanalysis \
  --primary-k 20 \
  --analysis-layer early \
  --bootstrap-iters 100 \
  --bootstrap-alpha 0.05 \
  --early-n 2 \
  --seed 42
```

Bu yol özellikle:
- threshold/policy denemeleri,
- calibration karşılaştırması,
- hızlı smoke test,
- stability suite öncesi temiz analiz tekrarları

için kullanışlıdır.

---

## 6) Parser Failure Bank

Parser/judge robustluğu için failure bank üret:

```bash
poetry run python scripts/build_parser_failure_bank.py \
  --input results/ablation_final/step_signal_table.csv \
  --out data/debug/parser_failure_bank.csv \
  --summary-out data/debug/parser_failure_bank_summary.json \
  --primary-k 20 \
  --max-rows 150
```

Beklenen:
- `data/debug/parser_failure_bank.csv`
- `data/debug/parser_failure_bank_summary.json`

Not:
- Eğer giriş tablosu eski şemadaysa `raw_step_text` boş kalabilir.
- Yeni koşularda step metni de saklandığı için bank gerçek regression verisi olarak kullanılabilir.

---

## 7) Demo Vaka Çıkarmak

Var olan sonuç klasöründen tek komutla vaka demo’su üret:

```bash
poetry run python scripts/run_demo_case.py \
  --results results/ablation_v3_full_1319_policy_layerfix \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --layer-selection early_warning_best \
  --out results/demo_case_deepseek_best_warning
```

Beklenen:
- `seismograph.html`
- `case_steps.csv`
- `case_summary.json`

Bu HTML içinde:
- warning score,
- threshold çizgisi,
- ilk alarm adımı,
- incorrect/parse-fail overlay’leri,
- final correctness etiketi

yer alır.

Faydalı opsiyonlar:
- `--dataset-index` vermezsen script ilginç bir hata örneğini otomatik seçer
- `--layer-selection fixed|classification_best|early_warning_best`
- `--analysis-layer early|middle|late` ile katmanı elle zorlayabilirsin

Not:
- Eski sonuç klasörlerinde `step_text` henüz persist edilmemiş olabilir. Bu durumda demo yine çalışır ama step hover içeriği boş kalır.

---

## 8) Stabilite Koşusu

Seed ve alt-split bazlı stabilite özeti için:

```bash
poetry run python scripts/run_stability_suite.py \
  --input results/ablation_v2_full_1319_fewshot/step_signal_table.csv \
  --out results/stability_suite \
  --primary-k 20 \
  --early-n 2 \
  --bootstrap-iters 100 \
  --bootstrap-alpha 0.05 \
  --seeds 7,42,123 \
  --slice-starts 0,400,800 \
  --slice-size 400
```

Beklenen:
- `stability_runs_long.csv`
- `stability_summary.csv`
- `stability_summary.json`

Headline kararını tek run yerine bu özet tablodaki `mean/std/min/max` üzerinden ver.

---

## 9) Rapor Figürleri

Tam koşudan rapor figürlerini üret:

```bash
poetry run python scripts/export_report_figures.py \
  --results results/ablation_v3_full_1319_policy_layerfix \
  --out-dir report/figures \
  --case-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --case-layer-selection early_warning_best
```

Beklenen figürler:
- `model_comparison_common_index.png`
- `model_comparison_best_layer.png`
- `k_sensitivity.png`
- `early_warning_threshold_sweep.png`
- `warning_trajectory.png`
- `case_seismograph.png`
- `report_summary.json`

Eger lokal Chrome/Kaleido koprusu yuzunden PNG export hata verirse:

```bash
poetry run python scripts/export_report_figures.py \
  --results results/ablation_v3_full_1319_policy_layerfix \
  --out-dir report/figures \
  --case-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --case-layer-selection early_warning_best \
  --html-only
```

Bu durumda en azindan `html` figürleri, manifest ve report summary dosyalari uretilir.

Raporu derlemek için:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 10) Sık Sorunlar ve Çözüm

- **`AUROC`/`AUPRC` = `NaN`**  
  Genelde tek sınıf çıkmıştır (hepsi doğru/yanlış).  
  Çözüm: `num-samples` artır, farklı `start-index` dene.

- **Koşu çok yavaş**  
  `max-new-tokens` 192 -> 128 yap; modeli ayrı ayrı çalıştır; `--torch-threads` ve `--torch-interop-threads` kombinasyonlarını dene.

- **`--cpu-int8` kullanınca `NoQEngine` hatası alıyorum**  
  Bu, mevcut PyTorch build’inde CPU quantization backend olmadığı anlamına gelir.  
  Çözüm: `--cpu-int8` olmadan devam et (script artık bu durumda otomatik FP32 fallback yapar).

- **Daha temiz progress/log istiyorum**  
  `--verbose --log-every <N>` kullan.  
  Varsayılan olarak tqdm progress bar var; yalnızca düz log istersen `--quiet-progress` ekle.

- **Step sinyali az çıkıyor**  
  Çözüm: `max-new-tokens` artır; step formatı bozuluyorsa prompt/generation ayarını sıkı tut.

- **Parser failure bank’te `raw_step_text` boş geliyor**  
  Büyük olasılıkla eski step tablosunu kullanıyorsun.  
  Çözüm: mevcut `run_ablation.py` ile yeni bir subset koş ve bank’i o çıktıdan üret.

- **Figure export sırasında PNG üretilmiyor**  
  Genelde `kaleido` eksiktir.  
  Çözüm: `poetry install` veya `poetry add kaleido`.

- **MPS/Memory hatası**  
  `GOR_DEVICE=cpu` ile devam et (şu an en stabil seçenek).

---

## 11) Başvuru Dosyası İçin “Minimum Güçlü Paket”

Finalde şunları rapora koy:
- A/B/C için tek toplu tablo (ana metrikler)
- 1-2 grafik (`experiment_b_model_comparison_common_index.html`, `experiment_c_*`)
- Negatif bulguları da yaz (özellikle `k` hassasiyeti ve parse-fail etkisi)

---

İstersen bir sonraki adımda sana **“final koşu komutlarını makinene göre optimize edilmiş”** tek bir runbook (kopyala-yapıştır) çıkarayım: kısa/orta/uzun profil şeklinde.