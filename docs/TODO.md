# TODO - Next Iteration Plan

Bu dosya, `results/ablation_v2_full_1319_fewshot/` kosusundan sonra bir sonraki iterasyon icin uygulanacak detayli plani toplar.

Ana hedef:
- Faz 4'u sadece "kosmus" degil, "guclu sekilde savunulabilir" hale getirmek.
- Faz 5'i demo + rapor + reproduksiyon tarafinda kapatmak.
- Erken uyariyi sadece ayrisabilir degil, gercekten erken hale getirmek.

## 1. Current Snapshot

- Full run tamamlandi: iki model de `1319/1319` ornek isledi.
- Experiment B common-index sayisi `1319`, yani model kiyasi ayni ornekler ustunde yapildi.
- Experiment A logistic model, baseline threshold'a gore belirgin ustun:
- `AUROC = 0.7613` vs `0.6548`
- `AUPRC = 0.5388` vs `0.4030`
- `Brier = 0.1989` vs `0.2665`
- `ECE = 0.1652` vs `0.2048`
- Erkenlik halen zayif:
- `alarm_before_error_rate = 0.1213`
- `lead_time_mean = -1.1121`
- `lead_time_median = -1.0`
- DeepSeek ayrisma gucunde Qwen'den daha iyi:
- DeepSeek `AUROC = 0.8223`, `Brier = 0.1584`
- Qwen `AUROC = 0.7107`, `Brier = 0.2176`
- Kalibrasyonda ise Qwen daha iyi:
- Qwen `ECE = 0.1186`
- DeepSeek `ECE = 0.2036`
- Parser/judge kalitesi halen iyilestirme istiyor:
- Qwen primary-k parse_fail step rate: `0.1515`
- Qwen primary-k no_math_signal step rate: `0.1181`
- DeepSeek primary-k parse_fail step rate: `0.1317`
- DeepSeek primary-k no_math_signal step rate: `0.0696`

## 2. Recommended Execution Order

Oncelik sirasini yalnizca "en onemli fikir"e gore degil, bagimliliklara gore kur:

1. Parser/judge robustlugu
2. Erkenlik optimizasyonu
3. Kalibrasyon katmani
4. Ek dogrulama ve stabilite raporu
5. Faz 5 demo/rapor kapatma

Neden bu sira:
- Parser kalitesi duzelmeden lead-time ve calibration iyilestirmeleri etiket gurultusune carpabilir.
- Erkenlik optimizasyonu, Faz 4'un asil arastirma boslugunu kapatir.
- Calibration, warning skorunu daha guvenilir yapar ama once skorun kendisi oturmali.
- Stabilite raporu, bu iyilestirmelerin sans olmadigini gosterir.
- Demo/rapor en sona birakilmali, cunku final hikaye son metriklerle kurulacak.

## 3. Fixed Decisions For This Iteration

Bu iterasyonda asagidaki noktalar sabit kalsin:

- Ana benchmark: `openai/gsm8k`, `main`, `test`
- Ana model cifti:
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `Qwen/Qwen2.5-1.5B-Instruct`
- Headline reporting icin primary `k = 20`
- Model karsilastirmasi icin primary scope = `common_index`
- Ana metrikler:
- Experiment A: `AUROC`, `AUPRC`, `Brier`, `ECE`, `lead_time`
- Experiment B: ayni metrikler + `signal_gap_mean`, `Cohen's d`
- Experiment C: `auroc_cv_over_k`, `Kendall tau`

Bu sabitleme, metrik drift ve p-hacking riskini azaltir.

## 4. Workstream A - Parser/Judge Robustlugu

### Goal

Ozellikle Qwen tarafinda `parse_fail` ve `no_math_signal` oranlarini dusurmek; ayni anda DeepSeek coverage kazaniminin korunmasi.

### Why This Matters

- Mevcut lead-time ve calibration analizi step-level reason etiketlerinin kalitesine bagli.
- `no_math_signal` ve `sympy_parse_error` oranlari, erken uyari skorunu gereksiz gurultulu hale getiriyor.
- Parser daha guclu hale gelirse Experiment A ve B yorumlari daha temiz olur.

### Files Likely Touched

- `src/evaluation/step_parser.py`
- `src/generation/extraction.py`
- `src/generation/runner.py`
- `src/evaluation/sympy_judge.py`
- `tests/test_step_parser.py`
- `tests/test_sympy_judge.py`

### Detailed Tasks

- `step_signal_table.csv` icinden hata bankasi cikart.
- Ilk hedef, Qwen primary-k satirlarindan `parse_fail`, `no_math_signal`, `sympy_parse_error`, `equation_mismatch` bucket'larindan ornekler toplamak.
- En az `100-150` representative step ornegi sec.
- Her ornek icin su alanlari bir debug tablosunda sakla:
- `model_name`
- `dataset_index`
- `step_index`
- `raw_step_text`
- `normalized_step_text`
- `current_reason`
- `expected_reason`
- `expected_parseable`

- Parser failure bank olustur.
- Yeni dosya onerisi: `data/debug/parser_failure_bank.csv`
- Bu dosya regression testlerinin veri kaynagi olacak.

- Normalization katmanini genislet.
- Para birimi, yuzde, binlik ayirac, parantezli aciklama, markdown kalintisi, extra punctuation, unit suffix, `x = ... therefore ...` gibi karma formatlari normalize et.
- Tek satir icinde birden fazla esitlik varsa ayrisma kurali ekle.
- `Final Answer:` satirinin step parser'i bozmadigindan emin ol.

- Parser adapter mantigi ekle.
- `Step n:` ile baslayan temiz format disinda su yapilari ayri handle et:
- `1) 40 - 8 = 32`
- `1. 40 - 8 = 32`
- `32, because 40 - 8 = 32`
- `40 - 8 = 32, so ...`
- dogrudan tek-step arithmetic cevaplar

- Reason taxonomy'yi daha net ayir.
- Mumkunse `no_math_signal`, `sympy_parse_error`, `unsupported_symbolic_form`, `boundary_detected_but_empty_math`, `non_equational_numeric_reasoning` gibi daha acik reason kodlari uret.
- Boylesi, hangi alt-problemin baskin oldugunu daha net gosterecek.

- Test kapsamini failure bank ile guclendir.
- Hata bankasindaki pattern'lerden regression testler ekle.
- Testler bir iki sentetik ornek yerine gercek fail pattern'lerini cover etsin.

### Deliverables

- `data/debug/parser_failure_bank.csv`
- Guncellenmis parser/judge kodu
- Yeni regression testleri
- Before/after parse reason histogram ozeti

### Success Criteria

- Qwen `parse_fail_step_rate <= 0.10`
- Qwen `no_math_signal_step_rate <= 0.06`
- DeepSeek `no_math_signal_step_rate` mevcut seviyenin ustune cikmasin
- `common_index_count` tam coverage olarak korunsun

### Validation Run

- Ilk validation: iki modelle `num_samples=200`
- Ikinci validation: tam full-run replay gerekmiyorsa `num_samples=400`
- Compare edilecek alanlar:
- `parse_fail_step_rate`
- `no_math_signal_step_rate`
- `sample_any_parse_fail_rate`
- `sample_any_no_math_rate`
- `common_index_count`

## 5. Workstream B - Erkenlik Optimizasyonu

### Goal

Warning skorunu yalnizca final failure'i ayristiran degil, hatadan once alarm veren bir sisteme yaklastirmak.

### Why This Matters

- Su an sistem "is bitince dogru tahmin eden" tarafa daha yakin.
- Arastirma iddiasi ise "reasoning derailment'i erken tespit etmek".
- Bu nedenle `lead_time` ve `alarm_before_error_rate`, bu iterasyonun en kritik ciktisi.

### Files Likely Touched

- `scripts/run_ablation.py`
- Gerekirse yeni modul: `src/experiments/early_warning.py`
- Gerekirse yeni plot util dosyalari

### Detailed Tasks

- Score ile alarm policy'yi birbirinden ayir.
- Mevcut logistic skorunu `warning_score` olarak sakla.
- Alarm verme karari ayri bir threshold/persistence policy katmani ile verilsin.

- Yeni early-warning policy ailesi dene.
- Static threshold
- Quantile-based threshold
- Z-score based threshold
- Delta threshold: bir onceki step'e gore ani sicrama
- Persistence policy: `k-of-m` veya ardisik `n` step alarm
- Hybrid policy: `warning_score + entropy jump`

- Threshold secimini yeni objective ile optimize et.
- Sadece AUROC/F1 tabanli secim yapma.
- Validation objective onerisi:
- `maximize alarm_before_error_rate`
- `subject to false_alarm_rate <= tau`
- veya agirlikli skor:
- `early_score = w1 * alarm_before_error_rate - w2 * false_alarm_rate + w3 * normalized_lead_time`

- Lead-time metrik ailesini genislet.
- `lead_time_mean`
- `lead_time_median`
- `alarm_before_error_rate`
- `first_alarm_step`
- `first_error_step`
- `false_alarm_before_any_error_rate`
- `late_alarm_rate`

- Plot ve rapor ciktilari ekle.
- Threshold sweep curve
- Alarm timing histogram
- Success vs failure icin step-index bazli ortalama warning score

- Cross-validation icinde leakage olmasin.
- Threshold/policy secimi her fold icinde train portion uzerinde yapilsin.
- Test fold sadece final evaluation icin kullanilsin.

### Deliverables

- Guncellenmis Experiment A outputs
- Yeni alarm policy comparison tablosu
- Yeni lead-time plotlari
- Yeni summary alanlari `ablation_summary.json` icinde

### Success Criteria

- `alarm_before_error_rate >= 0.30`
- `lead_time_mean >= -0.25`
- `lead_time_median >= 0` veya sifira anlamli sekilde yaklasma
- AUROC mevcut combined seviyeden `0.02`'den fazla dusmesin

### Validation Run

- Ilk hizli deneme: `num_samples=200`
- Ikinci deneme: full dataset replay
- Once eski policy vs yeni policy ayni run ustunde karsilastirilabilsin

## 6. Workstream C - DeepSeek Calibration Layer

### Goal

DeepSeek warning skorunun ECE'sini dusurmek ve olasilik yorumunu daha guvenilir hale getirmek.

### Why This Matters

- DeepSeek ayrisma gucunde cok iyi ama calibration zayif.
- Bunu duzeltirsek hem rapor guclenir hem de demo tarafinda "warning probability" daha guvenilir olur.

### Files Likely Touched

- `scripts/run_ablation.py`
- Gerekirse yeni modul: `src/evaluation/calibration.py`
- Gerekirse yeni plot util dosyalari

### Detailed Tasks

- OOF skor saklama yapisini netlestir.
- Calibration icin fold-disinda uretilmis `warning_score` lazim.
- Mevcut sample prediction ciktilarini calibration-compatible hale getir.

- Asagidaki post-hoc calibrator'lari dene:
- Platt scaling
- Isotonic regression
- Basit beta calibration opsiyonel olarak degerlendir

- Raw vs calibrated kiyas tablosu ekle.
- `AUROC`
- `AUPRC`
- `Brier`
- `ECE`
- reliability curve

- Calibration artifact kaydet.
- Model bazli calibration parametreleri bir JSON veya pickle-benzeri artifact olarak saklansin.
- Demo bu artifact'i okuyup calibrated probability gosterebilsin.

- Calibration'i yalnizca DeepSeek icin degil, opsiyonel olarak her iki model icin de calis.
- Headline hedef DeepSeek olsa da pipeline model-agnostic olursa daha temiz olur.

### Deliverables

- `raw_vs_calibrated` metric tablosu
- Reliability curve plotlari
- Model bazli calibrator artifact'lari

### Success Criteria

- DeepSeek `ECE <= 0.14`
- DeepSeek `Brier` kotulesmesin
- DeepSeek `AUROC` dususu `<= 0.01`

### Validation Run

- Once primary-k full run ciktilari uzerinde offline calibration dene
- Sonra gerekirse pipeline'a entegre et

## 7. Workstream D - Faz 5'i Kapatma

### Goal

Projeyi "arastirma kodu" seviyesinden "gosterilebilir demo + rapor" seviyesine tasimak.

### Why This Matters

- `docs/development_plan.md` Faz 5 dogrudan vitrinleme bekliyor.
- Su an `src/visualization/seismograph.py` ve `report/main.tex` var ama minimal.
- README'de tek-komutlu demo ve ornek vaka akisi henuz yok.

### Files Likely Touched

- `src/visualization/seismograph.py`
- Yeni script onerisi: `scripts/run_demo_case.py`
- `README.md`
- `report/main.tex`
- Yeni klasor onerisi: `report/figures/`

### Detailed Tasks

- Tek komut demo akisi kur.
- Hedef: mevcut results klasorunden bir sample secip warning curve + step table + final decision ureten tek komut.
- Ornek:
- `poetry run python scripts/run_demo_case.py --results ... --model ... --dataset-index ...`

- Iki showcase case sec.
- Bir "erken yakalanan fail" vakasi
- Bir "basarili cozum, dusuk warning" vakasi
- Bu iki vaka README ve raporda kullanilsin.

- Seismograph gorselini zenginlestir.
- Warning threshold cizgisi
- Alarm adimi dikey cizgi
- Step correctness overlay
- Final answer correctness bilgisi

- Summary figure seti hazirla.
- Model comparison bar chart
- Early warning threshold sweep / lead-time plot
- `k` sensitivity plot
- En az bir seismograph case-study figure

- `report/main.tex` genislet.
- Abstract
- Method
- Experimental setup
- Main results
- Failure analysis
- Limitations
- Next steps

- README reproduksiyon kismini finalize et.
- Setup
- Full run command
- Demo command
- Cikti dosyalari ne anlama geliyor bolumu

### Deliverables

- Tek komut demo scripti
- README demo akisi
- `report/main.tex` icin guncel rapor metni
- `report/figures/` altinda secilmis figurlar

### Success Criteria

- Harici biri README'yi izleyerek demo calistirabilsin
- `report/main.tex` compile edilebilsin
- README'de en az bir statik gorsel veya GIF olsun

## 8. Workstream E - Ek Dogrulama ve Stabilite

### Goal

Sonuclarin tek bir seed veya tek bir subset secimine bagli olmadigini gostermek.

### Why This Matters

- Faz 4 sonuclari iyi gorunuyor, ama reproduksiyon ve guvenilirlik icin stabilite raporu eksik.
- Parser, threshold ve calibration iyilestirmeleri geldikten sonra bu adim zorunlu.

### Files Likely Touched

- `scripts/run_ablation.py`
- Yeni script onerisi: `scripts/run_stability_suite.py`
- `docs/experiment_run_guide.md`

### Detailed Tasks

- Seed kontrolunu standart hale getir.
- Eger eksikse `--seed` argumani ekle.
- Bootstrap, CV split, threshold secimi ve calibration train/eval akisi bu seed'i kullansin.

- Alt-split protokolu tanimla.
- Onerilen ilk protokol:
- Slice A: `start_index=0`, `num_samples=400`
- Slice B: `start_index=400`, `num_samples=400`
- Slice C: `start_index=800`, `num_samples=400`
- Gerekirse son `119` ornek ayri raporlansin

- Multi-seed suite kos.
- Onerilen seed seti:
- `7`
- `42`
- `123`

- Stability summary tablosu uret.
- Her metric icin:
- mean
- std
- min/max
- count

- Report karar kuralini sabitle.
- Headline metric'ler icin "ortalama + std" ver.
- En iyi tek run yerine stabil run davranisini vurgula.

### Deliverables

- `scripts/run_stability_suite.py`
- Stability summary CSV/JSON
- Ek rapor bolumu veya appendix tablosu

### Success Criteria

- Combined Experiment A `AUROC` seed/slice bazinda dar bir bantta kalsin
- Ana kararlar model siralamasini degistirmesin
- Parser iyilestirmesi ve calibration sonrasi headline sonuc tersine donmesin

## 9. Milestone Plan

### Milestone 1 - Data Quality Lock

- Parser/judge robustlugu tamam
- Failure bank olusmus
- Qwen parse_fail ve no_math oranlari dusmus

### Milestone 2 - Early Warning Upgrade

- Yeni alarm policy raporlanmis
- Lead-time belirgin iyilesmis
- Experiment A yeni tablolar ve plotlarla guncel

### Milestone 3 - Reliable Warning Scores

- DeepSeek calibration eklenmis
- ECE anlamli sekilde dusmus
- Reliability curve raporlanmis

### Milestone 4 - Phase 5 Package

- Demo scripti hazir
- README akisi tamam
- Rapor figurlari hazir

### Milestone 5 - Stability Evidence

- Multi-seed/alt-split suite tamam
- Sonuclarin hangi kisimlarinin stabil oldugu dokumante edilmis

## 10. Suggested Short-Term Schedule

### Block 1

- Parser failure bank
- Normalization + parser adapter
- Regression testleri

### Block 2

- Early-warning policy refactor
- Threshold sweep
- Lead-time odakli objective

### Block 3

- Calibration layer
- Reliability plots
- Artifact export

### Block 4

- Demo command
- README finalization
- Report figures + text

### Block 5

- Stability suite
- Final summary tables
- Docs cleanup

## 11. Final Exit Criteria For This Iteration

Bu iterasyon bitmis sayilsin ancak su kosullar birlikte saglanirsa:

- Parser/judge hata oranlari belirlenen hedeflere inmis olsun
- Erkenlik metrikleri gozle gorulur sekilde iyilesmis olsun
- DeepSeek calibration metrikleri duzelmis olsun
- README uzerinden tek komut demo calisiyor olsun
- Rapor figurlari ve teknik anlatim tamam olsun
- Multi-seed/alt-split stabilite tablosu uretilmis olsun

Bu noktadan sonra proje, Faz 4 kapanmis ve Faz 5 vitrinleme seviyesine cikmis kabul edilebilir.
