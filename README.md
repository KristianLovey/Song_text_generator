
# Generiranje tekstova pjesama pomoću RNN (LSTM / GRU)

## Opis projekta
Ovaj projekt bavi se automatskim generiranjem tekstova pjesama na engleskom jeziku korištenjem rekurentnih neuronskih mreža (RNN). Implementirane su dvije arhitekture – **LSTM** i **GRU** – koje se treniraju nad korpusom tekstova pjesama te se uspoređuju prema perpleksnosti i subjektivnoj kvaliteti generiranog teksta. Cilj projekta je demonstrirati sposobnost neuronskih mreža da nauče jezične obrasce, strukturu stihova i stil glazbenih tekstova.

---

## Dataset
Za treniranje modela koriste se javno dostupni skupovi podataka s tekstovima pjesama:
- **Kaggle Lyrics Dataset** (razni izvođači i žanrovi)
- alternativno: tekstovi dohvaćeni putem **Genius API-ja**

Za testiranje i provjeru ispravnosti pipelinea uključen je mali **sintetički dataset**:
```

data/sample/lyrics_small.txt

```

Podaci su obrađeni na **character-level** razini, gdje svaki znak predstavlja jedan element ulazne sekvence.

---

## Struktura repozitorija
```

.
├── src/
│   ├── model_common.py      # Definicija LSTM / GRU neuronske mreže
│   ├── train_lstm.py        # Treniranje LSTM modela
│   ├── train_gru.py         # Treniranje GRU modela
│   ├── data_prep.py         # Čišćenje i tokenizacija podataka
│   ├── generate.py          # Generiranje novih tekstova pjesama
│   ├── eval.py              # Evaluacija modela (perpleksnost)
│   └── fetch_datasets.py    # Upute za dohvat datasetova
├── data/
│   ├── raw/                 # Sirovi tekstovi pjesama
│   ├── processed/           # Tokenizirani dataset
│   └── sample/              # Testni dataset
├── models/                  # Spremljeni modeli
├── outputs/                 # Generirani tekstovi
├── reports/                 # Izvještaj i prezentacija
├── requirements.txt
└── README.md

````

---

## Korištene metode i tehnologije
- Character-level modeliranje teksta  
- Embedding sloj za reprezentaciju znakova  
- Rekurentne neuronske mreže:
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
- Cross-entropy loss funkcija  
- AdamW optimizator  
- Dropout i gradient clipping  

---

## Instalacija
```bash
pip install -r requirements.txt
````

Preporučeno je koristiti Python 3.10+ i PyTorch s CUDA podrškom ako je dostupna.

---

## Predobrada podataka

```bash
python src/data_prep.py --input data/raw --output data/processed --seq-len 128
```

Rezultat predobrade su:

* `dataset.npz` – numerički kodirani tekst
* `vocab.json` – mapiranje znak ↔ indeks

---

## Treniranje modela

### LSTM

```bash
python src/train_lstm.py --data data/processed/dataset.npz --epochs 5
```

### GRU

```bash
python src/train_gru.py --data data/processed/dataset.npz --epochs 5
```

Trenirani modeli spremaju se u direktorij `models/`.

---

## Generiranje teksta

```bash
python src/generate.py \
  --checkpoint models/lstm_last.pt \
  --prompt "under neon" \
  --steps 400 \
  --temperature 1.1 \
  --top-k 50
```

Generirani tekst se sprema u `outputs/generated.txt`.

---

## Evaluacija

Modeli se evaluiraju pomoću **perpleksnosti**, koja mjeri koliko dobro model predviđa sljedeći znak:

```bash
python src/eval.py --checkpoint models/lstm_last.pt --data data/processed/dataset.npz
```

Niža perpleksnost označava bolju kvalitetu modela.

---

## Rezultati i zaključak

Dobiveni rezultati pokazuju da oba modela mogu generirati strukturirane tekstove pjesama. LSTM i GRU postižu usporedive rezultate, pri čemu GRU koristi manji broj parametara i brže konvergira. Projekt uspješno demonstrira primjenu dubokog učenja u kreativnom području obrade prirodnog jezika.

---

## Autor

Projekt izrađen u sklopu kolegija **Neuronske mreže**.


