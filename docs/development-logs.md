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
