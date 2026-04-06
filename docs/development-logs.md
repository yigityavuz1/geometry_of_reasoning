# Development Logs (Developer-Focused)

Bu dosya, projedeki her teknik gelistirmeyi standardize bir formatta kaydetmek icin kullanilir.
Amac: tekrar uretilebilirlik, karar izlenebilirligi ve hizli handoff.

## Log Kayit Standardi (Rubrik)

Her kayit asagidaki sirayi takip eder:

1. **Entry ID**
   - Format: `LOG-YYYYMMDD-XX` (aynı gun icin artan index)
2. **Context**
   - Kisa hedef: bu degisiklik neden yapildi?
3. **Scope**
   - Faz/alt-faz, etkilenen alanlar
4. **Files Touched**
   - Tam dosya yollari (repo-relative)
5. **Implementation Details**
   - Teknik olarak ne degisti? (modul/fonksiyon/sema)
6. **Decisions**
   - Alinan mimari veya arac secimi kararlarinin gerekcesi
7. **Validation**
   - Calistirilan test/lint/manual check ve sonucu
8. **Known Gaps / Risks**
   - Bilinen eksikler, gecici cozumler, riskler
9. **Next Actions**
   - Bir sonraki somut adimlar

## Yazim Kurallari

- "Ne yaptik" + "neden yaptik" birlikte yazilir.
- Belirsiz ifadelerden kacinilir ("duzeltildi" yerine tam teknik degisiklik yazilir).
- Metrik/sonuc varsa sayisal verilir.
- Her kayit, bagimsiz okunabilir olmalidir.
- TODO'lar eylem fiili ile baslar (or. "Implement", "Refactor", "Benchmark").

---

## Entries

### LOG-20260222-01

**Context**
- Proje icin calisir bir baslangic kod tabani olusturmak ve log standardini devreye almak.

**Scope**
- Altyapi bootstrap
- Poetry tabanli environment yonetimi
- Cekirdek modul/scrip test iskeleti

**Files Touched**
- `.gitignore`
- `pyproject.toml`
- `README.md`
- `configs/base.yaml`
- `scripts/run_generation.py`
- `scripts/run_judge.py`
- `scripts/run_metrics.py`
- `scripts/run_ablation.py`
- `src/__init__.py`
- `src/generation/__init__.py`
- `src/generation/runner.py`
- `src/generation/extraction.py`
- `src/metrics/__init__.py`
- `src/metrics/lid_estimators.py`
- `src/metrics/global_dim.py`
- `src/metrics/uncertainty.py`
- `src/evaluation/__init__.py`
- `src/evaluation/sympy_judge.py`
- `src/visualization/__init__.py`
- `src/visualization/seismograph.py`
- `tests/test_lid_math.py`
- `tests/test_sympy_judge.py`
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `notebooks/.gitkeep`
- `report/main.tex`
- `report/references.bib`
- `docs/development_plan.md`
- `docs/development-logs.md`

**Implementation Details**
- Poetry tabanli `pyproject.toml` olusturuldu.
- Temel generation akisi (`runner.py`) ve step boundary/entropy yardimcilari eklendi.
- Metrik motoru icin MLE, TwoNN, ABID proxy, PR ve entropy fonksiyonlari eklendi.
- SymPy tabanli step-level equational consistency judge eklendi.
- Seismograph icin Plotly tabanli temel figur ureticisi eklendi.
- Generation/judge/metrics/ablation icin CLI script iskeletleri olusturuldu.
- Unit test iskeleti (LID + SymPy judge) eklendi.
- `development_plan.md` icindeki env yonetimi tercihi `Poetry` olarak guncellendi.

**Decisions**
- Environment/package manager tercihi: `Poetry`.
- MVP asamasinda minimal ama calisir script-first yaklasimi secildi.
- ABID icin ilk adimda "practical proxy" eklendi; paper-faithful versiyon sonraki iterasyona birakildi.

**Validation**
- Dosya olusturma ve iskelet butunlugu manuel kontrol edildi.
- `python3 -m compileall src scripts tests` basarili (syntax-level check).
- IDE linter: yeni dosyalarda hata yok.

**Known Gaps / Risks**
- `sympy_judge.py` su an tam task-correctness degil, adim ici denklik tutarliligini test ediyor.
- `run_ablation.py` sadece scaffold, istatistik pipeline implement edilmedi.
- Hidden state extraction halen minimal; katman-seviyesi tam capture iterasyonu bekliyor.

**Next Actions**
- Implement: Poetry install + baseline test/lint kosulari.
- Implement: GSM8K ornekleri uzerinde generation-to-judge veri hatti.
- Refactor: ABID estimator'i paper-faithful formulasyona yaklastir.
- Implement: hidden-state layer hooks ve parquet output.

### LOG-20260222-02

**Context**
- Poetry ortamini gercekten calistirip tek GSM8K ornegi icin generation -> judge -> metrics zincirini end-to-end kosmak.

**Scope**
- Faz 1 (altyapi dogrulama) + Faz 2/3'e gecis niteliginde smoke calisma
- Script import stabilitesi
- Device secimi stabilitesi (Mac MPS riski)

**Files Touched**
- `pyproject.toml`
- `src/generation/extraction.py`
- `src/generation/runner.py`
- `src/metrics/lid_estimators.py`
- `scripts/run_generation.py`
- `scripts/run_judge.py`
- `scripts/run_metrics.py`
- `scripts/run_ablation.py`
- `scripts/run_smoke_gsm8k_pipeline.py`
- `README.md`
- `docs/development-logs.md`

**Implementation Details**
- Poetry kurulumunda platform uyumlulugu icin `bitsandbytes` Linux-marker ile sinirlandi.
- `run_smoke_gsm8k_pipeline.py` eklendi:
  - `openai/gsm8k` tek ornek yukleme
  - generation
  - step-level SymPy judge
  - token embedding kaydi
  - metrik ozet cikisi
- Generation trace'e token-embedding cikarma ve entropy ozeti eklendi.
- Scriptlerin tamamina repo-root `sys.path` bootstrap eklendi; `PYTHONPATH` mecburiyeti kaldirildi.
- LID kestiriminde sifir-mesafe kaynakli `NaN/inf` sorununa karsi numerik stabilizasyon eklendi.
- Cihaz secimi iyilestirildi:
  - Varsayilan: `cuda` varsa onu kullan, yoksa `cpu`
  - `GOR_DEVICE={cpu|cuda|mps}` ile manuel override
  - Apple MPS default'u kaldirildi (stabilite icin)

**Decisions**
- Smoke demonstrasyonu icin `sshleifer/tiny-gpt2` kullanildi (hizli dogrulama amacli).
- MPS'in buyuk modelde (`Qwen2.5-0.5B-Instruct`) `NDArray > 2**32` hatasi nedeniyle default secim CPU'ya cekildi.
- Daha temsilci smoke sonucu icin ek olarak `GOR_DEVICE=cpu` ile `Qwen/Qwen2.5-0.5B-Instruct` kosusu alindi.
- End-to-end smoke scripti, fazlar arasi entegrasyon dogrulama araci olarak kabul edildi.

**Validation**
- `poetry env use /opt/homebrew/bin/python3.11` + `poetry install` basarili.
- `poetry run python scripts/run_smoke_gsm8k_pipeline.py --model sshleifer/tiny-gpt2 ...` basarili.
- Uretilen artefaktlar:
  - `results/smoke_tiny/generation_trace.json`
  - `results/smoke_tiny/judged_trace.json`
  - `results/smoke_tiny/token_embeddings.npy`
  - `results/smoke_tiny/metrics_summary.json`
- Asama scriptleri ayri ayri da dogrulandi:
  - `scripts/run_judge.py` basarili
  - `scripts/run_metrics.py` basarili
- `GOR_DEVICE=cpu poetry run python scripts/run_smoke_gsm8k_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct ...` basarili.
- Qwen smoke metrik ozeti:
  - `lid_mle_mean`: `7.2459`
  - `twonn_global_id`: `4.6831`
  - `abid_mean`: `28.1844`
  - `participation_ratio`: `11.7677`
- Qwen smoke cikti dosyalari:
  - `results/smoke_qwen05_cpu/generation_trace.json`
  - `results/smoke_qwen05_cpu/judged_trace.json`
  - `results/smoke_qwen05_cpu/token_embeddings.npy`
  - `results/smoke_qwen05_cpu/metrics_summary.json`
- `python3 -m compileall src scripts` basarili.

**Known Gaps / Risks**
- Tiny model step-format uretmiyor; judge ciktisi anlamli matematiksel CoT yerine parse-fail verebiliyor.
- `abid_mean` degeri model/cikti kalitesine gore oynak; arastirma sunumu oncesi normalize/robust varyant gerekebilir.
- `Qwen2.5-0.5B-Instruct` MPS'te bellek limiti nedeniyle bu makinede dogrudan stabil degil.

**Next Actions**
- Implement: step boundary bazli gercek hidden-state toplama (token-level yerine step aggregation).
- Refactor: SymPy judge'i task-correctness seviyesine yaklastir (sadece equation consistency degil).
- Implement: Faz 2 icin parse-fail analizi ve normalize katmani.

### LOG-20260222-03

**Context**
- Faz 2'yi kapatmak icin step-level judge'i task-correctness'e yaklastirmak, normalize/parser katmani eklemek ve `data/processed/` altinda etiketli tablo + parse-fail raporu uretmek.

**Scope**
- Faz 2 (Step-level Judge ve Veri Hazirlama) tamamlama
- Judge heuristic refactor + parser/normalizer
- Processed veri tablosu ve parse-fail ozet raporu

**Files Touched**
- `src/evaluation/step_parser.py`
- `src/evaluation/sympy_judge.py`
- `src/generation/extraction.py`
- `scripts/run_judge.py`
- `scripts/run_smoke_gsm8k_pipeline.py`
- `scripts/run_phase2_labeling.py`
- `tests/test_sympy_judge.py`
- `tests/test_step_parser.py`
- `README.md`
- `docs/development-logs.md`
- `data/processed/gsm8k_step_labels.jsonl`
- `data/processed/gsm8k_step_labels.csv`
- `data/processed/gsm8k_step_labels_summary.json`

**Implementation Details**
- `src/evaluation/step_parser.py` eklendi:
  - matematik metin normalizasyonu (`normalize_math_text`)
  - step header temizleme (`strip_step_header`)
  - equation pair cikarimi (`extract_equation_pairs`)
  - numeric token cikarimi (`extract_numeric_tokens`)
  - GSM8K gold answer icin inline equation/final answer parserlari
- `src/evaluation/sympy_judge.py` genisletildi:
  - `TaskReference` yapisi (question values + target values + final answer)
  - `build_task_reference(...)` ile GSM8K gold referans cikarimi
  - `judge_step_task_correctness(...)` ile reference-aware heuristic judge
  - `summarize_judgement_records(...)` ile parse_fail/correct oranlari ve reason dagilimi
  - eski `judge_step_equational_consistency(...)` geriye uyumluluk icin korundu
- `split_steps(...)` davranisi duzeltildi:
  - `Step N:` oncesi preamble, adim listesine dahil edilmiyor
  - sadece gercek step bloklari donuluyor
- `scripts/run_judge.py` ve `scripts/run_smoke_gsm8k_pipeline.py`:
  - gold answer varsa task-correctness judge kullaniyor
  - `judge_summary` (parse_fail_rate, reason_counts vb.) yaziliyor
- `scripts/run_phase2_labeling.py` eklendi:
  - GSM8K altkumesi icin generation + step-level labeling zinciri
  - `data/processed/` altinda JSONL/CSV step table
  - parse-fail/correctness ozet JSON raporu
- Test kapsami artirildi:
  - task-correctness judge testleri
  - step parser/split davranis testleri

**Decisions**
- Task-correctness "strict proof checker" yerine "reference-aware heuristic judge" olarak tasarlandi:
  - equation mismatch -> direkt yanlis
  - equation parse edilemezse numeric fallback ile reference overlap kontrolu
  - parse edilemeyen ve math sinyali olmayan adimlar `parse_fail` olarak ayrik tutuldu
- Faz 2'de table formati icin pratiklik adina `JSONL + CSV` secildi.

**Validation**
- `poetry run pytest tests/test_sympy_judge.py tests/test_step_parser.py tests/test_lid_math.py` basarili (`11 passed`).
- Judge duzeltmesi sonrasi tekrar:
  - `poetry run pytest tests/test_sympy_judge.py tests/test_step_parser.py` basarili (`8 passed`).
- Faz 2 labeling kosusu:
  - `GOR_DEVICE=cpu poetry run python scripts/run_phase2_labeling.py --model Qwen/Qwen2.5-0.5B-Instruct --split test --start-index 0 --num-samples 2 --max-new-tokens 128 ...`
  - Basarili artefaktlar:
    - `data/processed/gsm8k_step_labels.jsonl`
    - `data/processed/gsm8k_step_labels.csv`
    - `data/processed/gsm8k_step_labels_summary.json`
  - Ozet metrikler (`num_samples=2`, `total_steps=7`):
    - `parse_fail_rate`: `0.2857`
    - `correct_rate`: `0.7143`
    - `correct_rate_non_parse_fail`: `1.0000`
    - reason dagilimi: `text_matches_reference_value=5`, `no_math_signal=2`
- Smoke regression:
  - `poetry run python scripts/run_smoke_gsm8k_pipeline.py --model sshleifer/tiny-gpt2 --split test --index 0 --out-dir results/smoke_tiny_phase2 --max-new-tokens 64 --k 10`
  - Basarili; `judged_trace.json` icinde `judge_summary` alanlari dogrulandi.

**Known Gaps / Risks**
- Task-correctness halen heuristic; alternatif dogru cozum yollarini eksik/yanlis siniflama riski var.
- Latex yogun adimlarda (ornegin `\frac`) SymPy parse basarisi degisken; su an numeric fallback ile telafi ediliyor.
- Faz 2 raporu kucuk altkume (`num_samples=2`) ile alindi; daha guclu istatistik icin daha buyuk orneklem gerekli.

**Next Actions**
- Benchmark: `num_samples` artirilarak parse_fail oraninin model ve prompt bazinda stabilitesi olculsun.
- Refactor: SymPy-Latex normalization (ozellikle `\frac`) iyilestirilerek parse_fail azaltilsin.
- Implement: Faz 3'e gecis icin step-boundary bazli hidden-state aggregation (layer snapshots + step-end pooling) devreye alinsin.

### LOG-20260222-04

**Context**
- Faz 3'te metrik motorunu tamamlamak: LID/ABID/TwoNN/PR/Entropy tarafini testli hale getirmek ve sentetik veride beklenen ID trendlerini dogrulamak.

**Scope**
- Faz 3 (Geometri/Belirsizlik Metrik Motoru)
- `k` hassasiyeti raporlama
- Sentetik trend validasyonu

**Files Touched**
- `src/metrics/lid_estimators.py`
- `scripts/run_metrics.py`
- `scripts/run_phase3_synthetic_validation.py`
- `tests/test_lid_math.py`
- `tests/test_uncertainty.py`
- `README.md`
- `docs/development-logs.md`
- `results/smoke_qwen05_cpu/metrics_summary_phase3.json`
- `results/phase3/synthetic_validation.json`

**Implementation Details**
- `src/metrics/lid_estimators.py` genisletildi:
  - girdi dogrulama (`_validate_input_matrix`)
  - otomatik `k` clipping (`_effective_k`)
  - ABID formulasyonu acisal varyans proxy'sinden, `mean(cos^2)` tabanli daha dogrudan ABID yaklasimina cekildi
  - `coefficient_of_variation(...)` ve `k_sweep_local_id(...)` eklendi
- `scripts/run_metrics.py` guclendirildi:
  - yeni arguman: `--k-values 5,10,20,40`
  - ciktiya `k_sweep`, `lid_mle_cv_over_k`, `abid_cv_over_k` alanlari eklendi
- `scripts/run_phase3_synthetic_validation.py` eklendi:
  - kontrollu subspace uretimi (intrinsic dim -> ambient dim)
  - estimatorlar icin trend kontrolu (ID arttikca olcumlerin artmasi)
  - her intrinsic boyut icin `k` hassasiyet (CV) raporu
- Test kapsami artirildi:
  - `tests/test_lid_math.py`:
    - ABID pozitiflik testi
    - low-vs-high intrinsic dim trend testi
    - `k_sweep` + CV testi
  - `tests/test_uncertainty.py`:
    - uniform logit entropy = `log(V)` testi
    - peaked distribution entropy dusus testi
    - numpy/torch entropy tutarlilik testi

**Decisions**
- Faz 3 dogrulamasinda "tam monotonic her adim" yerine "beklenen trend + k-hassasiyet raporu" yaklasimi secildi; boylece estimator bias'i ve kucuk-ornek oynakligi rapora yansitilabilir.
- K-sweep raporu mevcut `run_metrics.py` icine alindi (ayri script yerine) ki mevcut pipeline ile geriye uyum bozulmasin.

**Validation**
- Testler:
  - `poetry run pytest` basarili (`17 passed`).
- K-sweep metrik kosusu:
  - `poetry run python scripts/run_metrics.py --embeddings results/smoke_qwen05_cpu/token_embeddings.npy --k 10 --k-values 5,10,20,40 --out results/smoke_qwen05_cpu/metrics_summary_phase3.json`
  - Cikti ozet:
    - `lid_mle_cv_over_k`: `0.0383`
    - `abid_cv_over_k`: `0.0973`
- Sentetik Faz 3 validasyonu:
  - `poetry run python scripts/run_phase3_synthetic_validation.py --out results/phase3/synthetic_validation.json --intrinsic-dims 4,8,12 --k-values 5,10,20,40 --n-samples 500 --ambient-dim 64`
  - `trend_checks`:
    - `lid_mle_increasing_at_first_k: true`
    - `abid_increasing_at_first_k: true`
    - `twonn_increasing: true`
    - `pr_increasing: true`

**Known Gaps / Risks**
- ABID halen pratik bir lokal kestirim; paper-faithful tum varyantlar (ve acisal dagilim modelleme) sonraki iterasyonda ele alinabilir.
- Sentetik validasyon tek seed ve tek noise seviyesinde kosuldu; robustness icin cok-seed raporu eklenmeli.
- Metric engine su an token embedding girdisiyle calisiyor; step-boundary pooled hidden-state girdisine gecis bir sonraki teknik adim.

**Next Actions**
- Implement: step-boundary bazli hidden-state pooling (`mean`/`last`) ve `run_metrics.py` tarafinda step-level metrik girisi.
- Benchmark: Faz 3 sentetik validasyonu coklu seed + noise grid ile calistirip raporu genislet.
- Prepare: Faz 4 icin A/B/C ablation scriptlerine `k_sweep` ciktilarini bagla.

### LOG-20260222-05

**Context**
- Faz 4'e gecis: A/B/C ablation pipeline'ini scaffold seviyesinden calisir hale getirmek ve erken-uyari/model-karsilastirma/k-hassasiyeti ciktilarini uretmek.

**Scope**
- Faz 4 (Ana Deneyler - A/B/C)
- Step-level metrik tablosu uretimi
- Erken uyari + model karsilastirma + estimator hassasiyet analizleri

**Files Touched**
- `src/generation/extraction.py`
- `src/generation/runner.py`
- `scripts/run_ablation.py`
- `README.md`
- `docs/development-logs.md`
- `results/ablation_smoke_phase4_v2/ablation_summary.json`
- `results/ablation_smoke_phase4_v2/step_signal_table.csv`
- `results/ablation_smoke_phase4_v2/step_signal_table.jsonl`
- `results/ablation_smoke_phase4_v2/experiment_a_summary.json`
- `results/ablation_smoke_phase4_v2/experiment_a_sample_predictions.csv`
- `results/ablation_smoke_phase4_v2/experiment_a_step_scores.csv`
- `results/ablation_smoke_phase4_v2/experiment_b_model_comparison.csv`
- `results/ablation_smoke_phase4_v2/experiment_b_model_comparison.json`
- `results/ablation_smoke_phase4_v2/experiment_b_model_comparison.html`
- `results/ablation_smoke_phase4_v2/experiment_c_k_sensitivity.csv`
- `results/ablation_smoke_phase4_v2/experiment_c_k_sensitivity.json`
- `results/ablation_smoke_phase4_v2/experiment_c_k_sensitivity_Qwen_Qwen2.5-0.5B-Instruct.html`

**Implementation Details**
- `src/generation/extraction.py`:
  - `estimate_step_token_spans(...)` eklendi (char-boundary -> token-span tahmini)
- `src/generation/runner.py`:
  - `GenerationConfig`'e `collect_step_signals` eklendi
  - `generate_reasoning_trace(...)` model/tokenizer reuse destekleyecek sekilde genisletildi (model her ornekte yeniden yuklenmiyor)
  - step-level sinyal ciktilari eklendi:
    - `step_signal_rows`: `step_index`, `start_token`, `end_token`, `entropy_mean`, `embedding_mean`
    - `token_entropy`
- `scripts/run_ablation.py` scaffold'tan tam pipeline'a alindi:
  - Veri toplama: GSM8K subset + model generation + step judge + per-step geometri (`lid`, `abid`, `twonn`, `pr`) + entropy
  - Deney A (Early Warning):
    - ilk `N` adimdan prefix feature seti
    - cross-validated logistic (`final_failure` tahmini)
    - baseline threshold modeli
    - AUROC/AUPRC/Brier/ECE + lead-time hesaplari
  - Deney B (Model Comparison):
    - model bazli A metrikleri
    - warning-score ayrisimi (`signal_gap_mean`, Cohen's d, Mann-Whitney p-value)
  - Deney C (Estimator Sensitivity):
    - k bazli performans tablolari
    - `auroc_cv_over_k`
    - k'lar arasi ranking tutarliligi icin Kendall tau
  - Ciktilar: CSV/JSON + Plotly HTML grafikler
- `README.md` Faz 4 calistirma komutlariyla guncellendi.

**Decisions**
- Faz 4 metrik tablosu "input table verildiyse yukle, verilmezse generate et" seklinde tasarlandi; boylece hem hizli replay hem de uctan uca kosu destekleniyor.
- Final success etiketi:
  - birincil: generated trace icinde final answer match
  - fallback: tum step'ler dogru ve parse-fail yoksa success
- K-hassasiyeti hesaplari per-step token cloud uzerinde yapildi; bu sayede `k={5,10,20,40}` daha anlamli oluyor.

**Validation**
- Regresyon testi: `poetry run pytest` basarili (`17 passed`).
- Faz 4 smoke kosusu (Qwen, 3 sample):
  - `GOR_DEVICE=cpu poetry run python scripts/run_ablation.py --experiment ALL --models Qwen/Qwen2.5-0.5B-Instruct --split test --start-index 0 --num-samples 3 --max-new-tokens 192 --k-values 5,10 --primary-k 10 --early-n 2 --out results/ablation_smoke_phase4_v2`
  - Uretilen ana ciktilar:
    - `ablation_summary.json`
    - `step_signal_table.csv/jsonl`
    - `experiment_a_*`, `experiment_b_*`, `experiment_c_*`
  - Smoke ozet (kucuk-orneklem):
    - Experiment A logistic: `AUROC=0.50`, `AUPRC=0.6667`, `Brier=0.2222`
    - Experiment B signal gap: `1.0204`
    - Experiment C: `auroc_cv_over_k=0.0` (`k=5,10` icin ayni skor)

**Known Gaps / Risks**
- Kucuk orneklemde (n=3) metrikler yuksek varyansli; Faz 4 yorumlari icin genis subset kosusu gerekli.
- `step_token_spans` tahmini tokenizer-length tabanli approx; edge-case hizalama hatalari olabilir.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` ile model-karsilastirma smoke'u henuz kosulmadi (compute maliyeti).

**Next Actions**
- Benchmark: Faz 4'u daha buyuk subsette (`num_samples >= 50`) ve iki modelle (`DeepSeek-R1-Distill`, `Qwen-Instruct`) kos.
- Add: Experiment A/B/C icin confidence interval (bootstrap) katmani.
- Refactor: step-boundary token-span hizalamasini daha dogrudan char-token alignment ile guclendir.

### LOG-20260222-06

**Context**
- `results/ablation_v2_full_1319_fewshot/` sonrasi TODO iterasyonunu gercek repo durumuna tasimak: Faz 4'u savunulabilir hale getirmek ve Faz 5 icin demo/rapor/reproduksiyon paketini somutlastirmak.

**Scope**
- Parser/judge failure-bank altyapisi
- Early-warning policy refactor + timing odakli raporlama
- Calibration katmani
- Demo/report/stability tooling
- Dokumantasyon ve run-guide genisletme

**Files Touched**
- `scripts/run_ablation.py`
- `scripts/build_parser_failure_bank.py`
- `scripts/run_demo_case.py`
- `scripts/run_stability_suite.py`
- `scripts/export_report_figures.py`
- `src/experiments/early_warning.py`
- `src/evaluation/calibration.py`
- `src/visualization/seismograph.py`
- `README.md`
- `docs/experiment_run_guide.md`
- `report/main.tex`
- `tests/test_ablation_reporting.py`
- `tests/test_calibration.py`
- `tests/test_early_warning.py`
- `docs/TODO.md`
- `docs/development-logs.md`

**Implementation Details**
- `run_ablation.py` guclendirildi:
  - bootstrap CI (`AUROC`, `AUPRC`, `Brier`, `ECE`)
  - Experiment B icin `all_samples` + `common_index` ayri raporlama
  - `primary_k=20` metadata/rationale
  - per-model checkpoint/resume ve zengin step-table schema (`step_text`, `normalized_step_text`, `matched_values`)
- `src/experiments/early_warning.py` eklendi:
  - warning score ile alarm policy birbirinden ayrildi
  - static / quantile / z-score / persistence / hybrid policy ailesi eklendi
  - `alarm_before_error_rate`, `false_alarm_before_any_error_rate`, `late_alarm_rate`, `missed_alarm_rate`, `first_alarm_step_mean`, `first_error_step_mean` gibi timing odakli metrikler eklendi
- `src/evaluation/calibration.py` eklendi:
  - raw / Platt / Isotonic calibrator karsilastirmasi
  - reliability curve tablosu
  - model-bazli calibration artifact export
- Parser/judge failure-bank akisi eklendi:
  - `scripts/build_parser_failure_bank.py`
  - `data/debug/parser_failure_bank*_summary.json`
- Faz 5 tooling eklendi:
  - `scripts/run_demo_case.py` ile tek-vaka seismograph render
  - `scripts/export_report_figures.py` ile PNG + HTML figure export
  - `scripts/run_stability_suite.py` ile seed/slice stabilite ozeti
  - `src/visualization/seismograph.py` warning threshold, alarm marker, incorrect/parse-fail overlay ve final verdict desteğiyle genisletildi
- README, run-guide ve `report/main.tex` yeni arastirma hikayesine gore guncellendi.

**Decisions**
- Multi-model headline reporting icin primary scope = `common_index`.
- Headline `k` sabiti = `20`.
- Calibration pipeline model-agnostic tasarlandi; headline ihtiyac DeepSeek olsa da her model icin artifact uretilebiliyor.
- Demo, modeli yeniden kosturmak yerine mevcut results klasorunden vaka secerek render edecek sekilde tasarlandi; Faz 5 tekrar uretilebilirligi icin daha pratik.

**Validation**
- README ve `docs/experiment_run_guide.md` uzerinden tum yeni CLI'ler dokumante edildi.
- Yeni moduller icin test kapsami eklendi:
  - `tests/test_ablation_reporting.py`
  - `tests/test_calibration.py`
  - `tests/test_early_warning.py`
- 40-sample validation ve parser-failure-bank akisi calistirilip artefaktlar uretildi:
  - `results/ablation_validation_40_judgefix/`
  - `data/debug/parser_failure_bank_validation_40_judgefix.csv`
  - `data/debug/parser_failure_bank_validation_40_judgefix_summary.json`

**Known Gaps / Risks**
- `judgefix` semantik etiketi iyilestirse de warning score halen `parse_fail` sinyaline asiri duyarliydi; bu da discrimination metriklerinde gerileme yaratti.
- Parser hala belirli symbolic/algebraic phrasing pattern'lerinde kirli lhs uretebiliyordu (`x x`, `G: G`, `from Step 4 into Step 5`, vb.).
- Faz 5 paketi artik repo'da mevcut, ancak rapor metrikleri yeni parser/judge davranisiyla yeniden kosulmadan finalize edilmemeli.

**Next Actions**
- Fix: parser cleanup + symbolic-equation judging regresyonlari.
- Re-tune: warning score'unu `reason`-aware hale getir.
- Validate: 40-sample validation rerun ile `old good -> new bad` regresyonlarini sifirla.

### LOG-20260222-07

**Context**
- Kullanici istegiyle proje onerisi / plan / repo durumu yeniden karsilastirildi; judgefix sonrasi belirlenen parser regresyonlari ve daha buyuk mantik hatasi (symbolic equations'in `equation_mismatch` sayilmasi) duzeltildi.

**Scope**
- Parser cleanup for chained symbolic equations
- Judge logic fix for symbolic relations
- Reason-aware early warning score
- Development log realignment

**Files Touched**
- `src/evaluation/step_parser.py`
- `src/evaluation/sympy_judge.py`
- `src/experiments/early_warning.py`
- `scripts/run_ablation.py`
- `tests/test_step_parser.py`
- `tests/test_sympy_judge.py`
- `tests/test_early_warning.py`
- `docs/development-logs.md`

**Implementation Details**
- `src/evaluation/step_parser.py` guclendirildi:
  - `strip_step_header(...)` artik tekrarlanan `Step N:` basliklarini art arda siliyor
  - basit LaTeX normalize eklendi (`\frac`, `\times`, `\(`, `\)`, vb.)
  - inline `Step N` referanslari equation-side temizliginden once temizleniyor
  - colon-label / duplicate-symbol / descriptive-label cleanup eklendi:
    - `x x -> x`
    - `G: G -> G`
    - `Subtract 30 from both sides: 2S = 80` icin lhs `2S`
    - `Substituting ... from Step 4 into Step 5, we get A = ...` icin lhs `A`
- `src/evaluation/sympy_judge.py` mantik duzeltmesi:
  - parse edilebilir symbolic equations artik otomatik `equation_mismatch` olmuyor
  - `equation_mismatch` yalnizca gercek numeric contradiction durumlarina indirildi
  - boylece `0.75x = 19.50`, `G + S = 110`, `2S = 80`, `S + C + T = 20 + 80 + 160 = 260` gibi relation'lar desteklenebiliyor
- `src/experiments/early_warning.py` warning score refactor:
  - eski `+0.75 * parse_fail` agirligi yerine `reason`-aware weight table + daha kucuk `parse_fail` katkisi kullanildi
  - `equation_mismatch`, `sympy_parse_error`, `unsupported_symbolic_form` gibi reason'lar farkli siddette agirlik aliyor
  - `equation_matches_reference`, `equation_consistent_supported` gibi acik destek sinyalleri warning score'u asagi cekebiliyor
- `scripts/run_ablation.py` summary metadata yeni warning-score spesifikasyonuyla hizalandi.

**Decisions**
- Bu iterasyonda en buyuk mantik hatasi olarak "symbolic equation != solved equality" yanilgisi kabul edildi ve judge mantigi buna gore yeniden sinirladi.
- `no_math_signal` artik tek basina agresif warning artisi yaratmiyor; daha belirleyici `reason` kodlari one alindi.
- Faz 4/Faz 5 hikayesini bozmayacak minimum ama etkili fix secildi; multi-layer hidden-state redesign bu turda kapsama alinmadi.

**Validation**
- Hedefli test kosusu:
  - `poetry run pytest tests/test_step_parser.py tests/test_sympy_judge.py tests/test_early_warning.py tests/test_ablation_reporting.py tests/test_calibration.py`
  - Sonuc: `34 passed`
- Manuel spot-check:
  - `0.75x = 19.50` -> `equation_matches_reference`
  - `S + C + T = 20 + 80 + 160 = 260` -> `equation_matches_reference`
  - `Subtract 30 from both sides: 2S = 80` -> `equation_consistent_supported`
- Offline reason-aware reanalysis (`results/ablation_validation_40_judgefix/step_signal_table.csv`) sonucu:
  - `AUROC = 0.7624`
  - `AUPRC = 0.6429`
  - `Brier = 0.1891`
  - `ECE = 0.1321`
  - `lead_time_mean = -0.0667`
  - `selected_alarm_policy = static_q75`
  - Bu, judgefix sonrasi dusen discrimination metriklerinin warning-score refactor ile buyuk olcude toparlanabildigini gosteriyor.

**Known Gaps / Risks**
- Parser/judge fix'lerinin tam etkisini gormek icin yeni 40-sample rerun hala gerekli; mevcut offline reanalysis eski step-table etiketlerini kullanir.
- Lead-time hala arastirma hedefinin gerisinde; discrimination toparlansa da erkenlik problemi tam kapanmis degil.
- Proje onerisi/Faz 3 hedefine kiyasla hidden-state tarafi halen esasen final-layer + approx step-token span uzerinden gidiyor; erken/orta/gec layer snapshot hedefi ileriki turda acilabilir.

**Next Actions**
- Run: yeni parser/judge/warning-score ile 40-sample validation rerun.
- Build: ayni run icin yeni parser failure bank.
- Decide: regresyonlar sifirlanir ve metric toparlanmasi korunursa 200/400/full replay'e cik.

### LOG-20260222-08

**Context**
- `ablation_validation_200_residualfix/` analizi sonrasinda kritik blokaj olarak sample coverage/accounting kaybi tespit edildi: bazi trace'ler step tablosundan tamamen dusuyor, dolayisiyla Experiment A/B ornek sayilari yapay olarak azaliyordu. Ayni anda parser failure bank'te kalan yuzde-gosterimi, birim donusumu ve self-correction karisikligi gibi residual issue'lar da temizlenmeliydi.

**Scope**
- Faz 4 validasyon muhasebesi / coverage fix
- Sample-level scoring aggregator guclendirme
- Residual parser/judge edge-case temizligi
- Validation ve log hizalama

**Files Touched**
- `scripts/run_ablation.py`
- `scripts/build_parser_failure_bank.py`
- `src/experiments/early_warning.py`
- `src/evaluation/step_parser.py`
- `src/evaluation/sympy_judge.py`
- `tests/test_ablation_reporting.py`
- `tests/test_early_warning.py`
- `tests/test_sympy_judge.py`
- `docs/development-logs.md`

**Implementation Details**
- `scripts/run_ablation.py` coverage/accounting tarafi guclendirildi:
  - `step_tokens.shape[0] < 3` olan step'ler artik tamamen drop edilmiyor; kisa step'ler entropy/judge bilgisiyle tabloya yaziliyor, geometri alanlari gerektiğinde `NaN` kalabiliyor.
  - Hic usable step-row uretilemeyen sample'lar icin fallback satirlari eklendi:
    - `trace_format_fail`
    - `trace_signal_fail`
    - `empty_completion`
  - Bu fallback satirlari `step_index=0` ile sample'in evaluation coverage'ini koruyor; sample artik progress'te "tamamlandi" olup tabloda gorunmez olmuyor.
  - Geometri ozetleri icin `_metric_summary_for_tokens(...)` yardimcisi eklendi; `PR` icin `n>=2`, LID/ABID/TwoNN icin `n>=3` sartlari ayrik ele alindi.
- Sample-level feature builder (`_build_sample_features`) zenginlestirildi:
  - Eski `prefix_mean` odakli feature seti genisletildi.
  - Yeni feature'lar:
    - `warning_prefix_max`
    - `warning_prefix_last`
    - `warning_prefix_top2_mean`
    - `warning_delta_prefix_max`
    - `hybrid_warning_prefix_max`
    - `reason_weight_prefix_mean/max`
    - `parse_fail_prefix_any`
    - `trace_failure_prefix_any`
    - `observed_prefix_steps`, `observed_step_rows`
  - Boylesiyle step-level guclu ama sample-levelde zayif kalan DeepSeek benzeri davranislar icin "erken pik / sert sicrama / fallback trace" sinyalleri kaybolmuyor.
  - Model girisine gitmeden once `_prepare_feature_matrix(...)` ile `NaN/inf` feature'lar median-fill + zero-fill stratejisiyle stabil hale getirildi.
- `src/experiments/early_warning.py`:
  - Yeni reason weight'ler eklendi: `trace_format_fail`, `trace_signal_fail`, `empty_completion`, `missing_step_judgement`.
  - `prepare_warning_features(...)` artik eksik `lid/entropy` alanlarini imputasyonla ele aliyor; warning score `NaN` uretmiyor.
- `src/evaluation/step_parser.py` residual edge-case cleanup:
  - Basit imperial unit normalization eklendi:
    - `miles -> feet`
    - `feet + inches -> feet`
    - `inches -> feet`
  - Boylece `4 feet - 6 inches = 3 feet 6 inches = 42 inches` ve `3 miles = 3 * 5280 = 15840 feet` tarzi step'ler parser tarafinda daha temiz temsil edilebiliyor.
  - Self-correction prose icin ek split kuralı eklendi:
    - `Wait ...`
    - `But wait ...`
    - `Hold on ...`
    - `Oops ...`
  - `_finalize_cleaned_side(...)` icindeki "rightmost operator fragment" heuristic'i yalnizca gercek descriptive-text iceren LHS'lerde uygulanacak sekilde daraltildi; saf aritmetik ifadelerin basi artik yanlislikla kirpilmiyor.
- `src/evaluation/sympy_judge.py`:
  - Yuzde-gosterimi ile decimal-oran arasindaki residual mismatch'ler icin `_expr_percent_display_match(...)` eklendi.
  - Boylece `0.3333 * 100 = 33.33%` gibi display-format esitlikleri dogru kabul edilebiliyor.
- `scripts/build_parser_failure_bank.py`:
  - Varsayilan reason bucket listesi yeni accounting reason'lariyla genisletildi:
    - `trace_signal_fail`
    - `trace_format_fail`
    - `empty_completion`
  - Boylece validation rerun sonrasi olusan fallback trace'ler debug bank'te manuel review icin gorunur kalacak.

**Decisions**
- Coverage problemi "step yoksa sample yok" seklinde cozulmek yerine, trace-level failure durumlarini explicit reason kodlariyla tabloda temsil eden bir muhasebe tasarimina cekildi.
- Sample-level scorer icin model sinifi degistirilmedi; proje planindaki "hafif logistic / threshold-based" yaklasim korunarak feature seti zenginlestirildi.
- Fake geometry uretmek yerine eksik geometri alanlari kontrollu imputasyonla downstream score hesabina tasindi; bu sayede parser/accounting fail'leri saklanmadi.

**Validation**
- Hedefli regresyon paketi:
  - `poetry run pytest tests/test_step_parser.py tests/test_sympy_judge.py tests/test_early_warning.py tests/test_ablation_reporting.py`
  - Sonuc: `46 passed`
- Tam test paketi:
  - `poetry run pytest`
  - Sonuc: `57 passed`
- CLI smoke:
  - `poetry run python scripts/run_ablation.py --experiment A --models sshleifer/tiny-gpt2 --split test --start-index 0 --num-samples 1 --max-new-tokens 48 --k-values 5 --primary-k 5 --early-n 2 --bootstrap-iters 10 --out results/ablation_smoke_accountingfix_tiny`
  - Sonuc: basarili; `ablation_summary.json` ve per-model `step_signal_table.csv` yazildi.
- In-memory sentinel smoke:
  - `_run_experiment_a_core(...)` uzerinde `trace_format_fail` satiri iceren minimal tablo kosuldu.
  - Sonuc: `coverage.trace_failure_samples = 1`, `coverage.zero_step_samples = 1`, pipeline hata vermeden sample-level output uretti.
- IDE linter:
  - Editlenen dosyalarda hata yok.

**Known Gaps / Risks**
- Bu iterasyon coverage ve residual parsing blokajlarini temizliyor; fakat gercek metrik etkisini gormek icin yeni 200-sample rerun hala zorunlu.
- Multi-layer hidden-state capture (`L/4`, `L/2`, `L`) hala planin gerisinde; Faz 3/Faz 4 hizalamasinin bir sonraki buyuk yapisal isi olarak duruyor.
- Sample-level scorer daha guclu olsa da halen hand-crafted feature + logistic yapisinda; tam replay sonrasinda per-model calibration/feature importance incelemesi gerekebilir.

**Next Actions**
- Run: yeni accounting + scoring + parser/judge fix'leriyle 200-sample validation rerun.
- Build: ayni kosu icin parser failure bank'i yeni reason kodlari (`trace_*`) dahil edilerek yeniden uret.
- Compare: yeni kosuyu `ablation_validation_200_residualfix/` ile coverage, AUROC/AUPRC, lead-time ve common-index bazinda karsilastir.
- Decide: metrikler stabilse full replay'e gec; degilse siradaki yapisal adim olarak multi-layer hidden-state capture'i ac.

### LOG-20260222-09

**Context**
- `docs/development_plan.md` ile repo arasindaki en buyuk yapisal acik kapanmamis durumda duruyordu: hidden-state toplama hala fiilen tek katman (`late`) uzerinden gidiyordu. Ayni anda prompt protokolunde literal placeholder kalintisi oldugu icin model bazen template'i kopyaliyor, unit normalization ise tum normalize hattina global uygulandigi icin gereksiz yan etkilere acik hale geliyordu.

**Scope**
- Prompt placeholder contradiction temizligi + retry mekanizmasi
- Unit normalization'i denklem-odakli parse hattina tasima
- Multi-layer hidden-state capture (`early/middle/late`) ve downstream Experiment A/B/C uyarlamasi
- Validation ve smoke hizalama

**Files Touched**
- `src/generation/runner.py`
- `src/evaluation/step_parser.py`
- `src/evaluation/sympy_judge.py`
- `scripts/run_ablation.py`
- `scripts/run_generation.py`
- `scripts/run_smoke_gsm8k_pipeline.py`
- `tests/test_generation_runner.py`
- `tests/test_step_parser.py`
- `tests/test_ablation_reporting.py`
- `docs/development-logs.md`

**Implementation Details**
- `src/generation/runner.py`
  - Prompt protokolunden literal placeholder satiri cikarildi; `Final Answer: <single number>` yerine gercek sayili format ornegi kullanildi.
  - Model ciktisinda acik template-kopyasi gorulurse (`<single number>`, `<equation>`, `Step N: <...>`, `Final Answer: <...>`) bir adet strict retry yapilacak sekilde generation akisi guncellendi.
  - `GenerationConfig` genisletildi:
    - `capture_layer_names`
    - `retry_on_placeholder_output`
    - `max_format_retries`
  - Hidden states artik tek final layer yerine secili snapshot seti icin toplanıyor:
    - `early`
    - `middle`
    - `late`
  - Geriye uyumluluk korundu:
    - `token_embeddings` halen `late` layer alias'i olarak yaziliyor.
    - `step_signal_rows` halen `late` layer alias'i olarak yaziliyor.
  - Yeni trace alanlari:
    - `captured_layers`
    - `token_embeddings_by_layer`
    - `step_signal_rows_by_layer`
    - `format_retry_count`
    - `format_retry_issue`
- `src/evaluation/step_parser.py`
  - `normalize_math_text(...)` icin `convert_units` bayragi eklendi.
  - Boylece miles/feet/inches donusumu artik tum normalize hattina global uygulanmiyor.
  - Imperial-unit donusumu yalnizca denklem-side temizleme asamasinda aktif.
- `src/evaluation/sympy_judge.py`
  - `_safe_sympify(...)` unit-aware parse yolunu kullanacak sekilde guncellendi; judge tarafinda equation-specific unit chain destegi korundu.
- `scripts/run_ablation.py`
  - Step tablo semasi multi-layer olacak sekilde genisletildi:
    - `layer_name`
    - `layer_index`
  - Generation sonrasi her step/k cifti icin artik her capture layer adina ayri satir uretiliyor.
  - Fallback/sentinel satirlari (`trace_*`, `empty_completion`) katman bazinda da yaziliyor; coverage mantigi bozulmadi.
  - Yeni `--analysis-layer` argumani eklendi; Experiment A/B/C secilen layer uzerinden kosuyor.
  - Root summary'ye `available_layers` ve `analysis_layer` yaziliyor.
  - Experiment A icin yeni katman karsilastirma artefakti eklendi:
    - `experiment_a_layer_comparison.csv`
    - `experiment_a_layer_comparison.json`
  - `Experiment B` ve `Experiment C` satirlari secili `analysis_layer` bilgisini tasiyor.
- `scripts/run_generation.py` ve `scripts/run_smoke_gsm8k_pipeline.py`
  - Buyuk JSON izlerini sistirmemek icin raw multi-layer embedding payload'i trace JSON'dan ayri dosyalara yaziliyor.
  - Smoke script artik katman-bazli embedding dosyalari ve `metrics_by_layer` ozeti de uretiyor.

**Decisions**
- Multi-layer geciste eski kodu bozmamak icin `late` layer alanlari backward-compatible alias olarak korundu; boylece mevcut analiz ve yardimci script'ler aniden kirilmadi.
- Unit normalization tamamen silinmedi; ama global text normalization yerine yalnizca equation parse yoluna daraltildi. Bu, hem `imperial conversion` edge-case'ini koruyor hem de soru/metin parse yan etkilerini azaltiyor.
- Placeholder retry mantigi yalnizca acik template-copy pattern'leri icin tetikleniyor; boylece normal matematiksel `<` / `>` kullanimlari gereksiz retry'a sebep olmuyor.

**Validation**
- Hedefli regresyon paketi:
  - `poetry run pytest tests/test_generation_runner.py tests/test_step_parser.py tests/test_sympy_judge.py tests/test_ablation_reporting.py tests/test_early_warning.py tests/test_calibration.py`
  - Sonuc: `55 passed`
- Tam test paketi:
  - `poetry run pytest`
  - Sonuc: `64 passed`
- Multi-layer smoke:
  - `poetry run python scripts/run_ablation.py --experiment A --models sshleifer/tiny-gpt2 --split test --start-index 0 --num-samples 1 --max-new-tokens 48 --k-values 5 --primary-k 5 --early-n 2 --bootstrap-iters 10 --analysis-layer early --out results/ablation_smoke_multilayer_early`
  - Sonuc:
    - basarili
    - `available_layers = ["early", "late", "middle"]`
    - `rows = 3` (1 sample x 1 step x 3 layer)
    - `analysis_layer = "early"`
- IDE linter:
  - Editlenen dosyalarda hata yok.

**Known Gaps / Risks**
- Yeni multi-layer schema ile en temiz kosu davranisi icin yeni bir `--out` dizini kullanmak daha guvenli; eski late-only checkpoint'lerle ayni output path uzerinde resume yapmak tavsiye edilmiyor.
- Multi-layer capture halen tam hook-temelli hafif bir extraction degil; mevcut implementasyon secili snapshot'lari cikarmak icin `output_hidden_states=True` forward pass'i kullaniyor.
- Experiment A layer-comparison artefakti eklendi; ancak daha genis "layer x model" anlatisi icin 200-sample validation sonrasinda rapor/sunum katmaninda ikinci bir toplu ozet daha gerekebilir.

**Next Actions**
- Run: yeni output klasorunde 200-sample validation'i multi-layer capture ile tekrar kostur.
- Build: ayni kosu icin parser failure bank'i yeniden uret.
- Compare:
  - `late` layer'i onceki accounting-fix kosusuyla karsilastir
  - `early/middle/late` layer comparison artefaktini incele
- Decide: en iyi erkenlik/dogruluk trade-off'unu veren layer'i full replay default'u olarak sabitle.

### LOG-20260222-10

**Context**
- Multi-layer validation sonrasi kalan 4 blokaji kapatmak:
  - residual parser/judge false-positive aileleri
  - layer-aware raporlama
  - reproducibility metadata/parquet eksigi
  - stale repo dokumani/rapor anlatimi

**Scope**
- Parser/extraction cleanup
- Reproducibility ve output schema
- Experiment A/B layer-aware reporting
- README / run guide / report hizalama

**Files Touched**
- `pyproject.toml`
- `poetry.lock`
- `src/generation/extraction.py`
- `src/evaluation/step_parser.py`
- `src/generation/runner.py`
- `scripts/run_ablation.py`
- `scripts/run_generation.py`
- `scripts/run_smoke_gsm8k_pipeline.py`
- `tests/test_step_parser.py`
- `tests/test_sympy_judge.py`
- `tests/test_ablation_reporting.py`
- `README.md`
- `docs/experiment_run_guide.md`
- `report/main.tex`
- `docs/development-logs.md`

**Implementation Details**
- `src/generation/extraction.py`
  - `Final Answer:` ve `Step N:` marker'lari ayni satira yapistiginda split edilebilsin diye inline-marker ayirma eklendi.
  - `Final Answer:` sadece gercekten sonrasinda yeni step header yoksa truncate edilecek sekilde daraltildi; boylece `Final Answer: ... Step 1: ...` turu bozuk ama kurtarilabilir trace'ler tamamen kaybolmuyor.
- `src/evaluation/step_parser.py`
  - LHS tarafinda saf anlatim etiketi olan cok-kelimeli phrase'leri (`Total cost of lemons` gibi) equation zincirinin parcası sanmayan yeni heuristik eklendi.
  - Tek harfli sembol assignment'lari (`S = 20`, `x = ...`) korunacak sekilde heuristik dar tutuldu; mevcut correct symbolic flows bozulmadi.
- `src/generation/runner.py`
  - `collect_model_metadata(...)` eklendi.
  - Generation trace artik `model_metadata` tasiyor:
    - resolved model name
    - model/tokenizer revision
    - device
    - dtype
    - quantization/compile flags
- `scripts/run_ablation.py`
  - Ana tablolar icin ortak `csv + parquet` yazma yardimcisi eklendi.
  - `--input` artik `.parquet` da kabul ediyor.
  - Root outputlara `run_metadata.json` eklendi; `ablation_summary.json` icine `reproducibility` blogu ve `config_hash` yaziliyor.
  - Step table output layout'i artik root/combined/model seviyesinde parquet yollarini da iceriyor.
  - Experiment A layer comparison'dan otomatik `layer_recommendation` uretiliyor (AUROC birincil, timing tie-break).
  - Experiment B'ye yeni scope eklendi:
    - `experiment_b_model_comparison_best_layer.*`
    - her modeli kendi en iyi dogrulanan katmaninda raporluyor
    - sabit `common_index` fair-comparison ana raporu korunuyor
- `scripts/run_generation.py` ve `scripts/run_smoke_gsm8k_pipeline.py`
  - `model_metadata.json` ayri artefakt olarak da yaziliyor.
- Repo docs
  - README ve run guide parquet, `run_metadata.json`, `model_metadata.json`, `experiment_a_layer_comparison`, `experiment_b_model_comparison_best_layer` ciktilarini anlatacak sekilde guncellendi.
  - `report/main.tex` eski single-layer / stale full-run anlatimindan cikartilip mevcut multi-layer validation durumuna cekildi.
- Dependency
  - `pyarrow` Poetry dependency olarak eklendi; parquet output artik repo seviyesinde resmi bir capability.

**Decisions**
- `Experiment B` primary scope'u degistirilmedi; best-layer view eklendi ama fair fixed-layer common-index raporu ana referans olarak korundu.
- Parser tarafi yalnizca belirgin false-positive ailelerini hedefleyecek kadar dar tutuldu; gercek reasoning hatalarini "correct"e cevirecek agresif kurtarma uygulanmadi.
- Reanalysis (`--input`) modunda model revision ancak mevcutsa preserve ediliyor; ciplak eski CSV tablolar icin model metadata minimum model-name seviyesinde kalabilir.

**Validation**
- Targeted regression:
  - `poetry run pytest tests/test_step_parser.py tests/test_sympy_judge.py tests/test_ablation_reporting.py tests/test_generation_runner.py tests/test_early_warning.py tests/test_calibration.py`
  - Sonuc: `60 passed`
- Offline smoke (input-table path):
  - `poetry run python scripts/run_ablation.py --experiment ALL --input results/_tmp_ablation_all_smoke/step_signal_table.csv --primary-k 20 --analysis-layer late --bootstrap-iters 10 --early-n 2 --out results/_tmp_ablation_repro_smoke`
  - Sonuc:
    - `ablation_summary.json` yazildi
    - `run_metadata.json` yazildi
    - `experiment_b_model_comparison_best_layer.csv` yazildi
    - root/combined parquet output path'leri summary'ye girdi

**Known Gaps / Risks**
- `--input` ile sadece bare step table verildiginde root `model_metadata.json` dosyalari eski kaynaktan tasinmiyorsa revision bilgisi minimum seviyede kalabilir; tam reproducibility icin ayni run'in `models/*/model_metadata.json` dosyalari da korunmali.
- Inline marker ayirma su an `split_steps(...)` seviyesinde; bozuk trace'lerde geometri boundary hizasi degil, en azindan parser/accounting kurtarimi hedefleniyor.
- `report/main.tex` artik stale olmaktan cikti ama final full replay metrikleri degil, mevcut 200-sample multi-layer validation perspektifini anlatiyor.

**Next Actions**
- Run: guncel kodla yeni bir `200`-sample multi-layer validation al.
- Build: ayni kosu icin parser failure bank'i yeniden uret.
- Inspect:
  - `experiment_a_layer_comparison.csv`
  - `experiment_b_model_comparison_best_layer.csv`
  - `run_metadata.json`
- Decide: final replay icin sabit `analysis_layer` mi, yoksa reporting'de fixed-layer + best-layer cift gorunumu mu headline olacak, bunu dondur.
