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
