# Generiranje tekstova pjesama pomoću RNN (LSTM / GRU)

## Opis projekta
Ovaj projekt bavi se automatskim generiranjem tekstova pjesama na engleskom jeziku korištenjem rekurentnih neuronskih mreža (RNN). U sklopu projekta implementirana je **vlastita neuronska mreža** temeljena na arhitekturama **LSTM** i **GRU**, bez korištenja gotovih modela za generiranje teksta. Modeli se treniraju na velikom korpusu tekstova pjesama te se uspoređuju prema perpleksnosti i subjektivnoj kvaliteti generiranog teksta.

Cilj projekta je demonstrirati sposobnost RNN modela da nauče jezične obrasce, strukturu stihova i stil glazbenih tekstova na razini znakova (character-level).

---

## Dataset
Za treniranje modela korišten je javno dostupan dataset:

- **Song Lyrics Dataset (Kaggle)**  
  Autor: *deepshah16*  
  Sadrži tekstove pjesama različitih izvođača i žanrova.

Iz izvornog dataseta izgrađen je jedinstveni tekstualni korpus:

```

data/raw/corpus.txt

```

Korpus je dobiven spajanjem svih tekstova pjesama u jednu datoteku te služi kao ulaz u proces predobrade.

Podaci se obrađuju na **character-level** razini, gdje svaki znak predstavlja jedan element ulazne sekvence.

---

## Struktura repozitorija
```

.
├── src/
│   ├── model.py          # Vlastita implementacija LSTM / GRU neuronske mreže
│   ├── train.py          # Treniranje modela (LSTM ili GRU)
│   ├── data_prep.py      # Predobrada: iz corpus.txt u dataset.npz + vocab.json
│   ├── generate.py       # Generiranje novih tekstova pjesama
│   └── eval.py           # Evaluacija modela (perpleksnost)
├── data/
│   ├── raw/
│   │   └── corpus.txt    # Sirovi tekstualni korpus
│   └── processed/
│       ├── dataset.npz   # Numerički kodiran tekst
│       └── vocab.json    # Mapiranje znak ↔ indeks
├── models/               # Spremljeni trenirani modeli
├── outputs/              # Generirani tekstovi
├── projekt.ipynb         # Notebook za rad s datasetom
└── README.md

````

---

## Korištene metode i tehnologije
- Character-level modeliranje teksta  
- Embedding sloj za reprezentaciju znakova  
- Rekurentne neuronske mreže:
  - **LSTM (Long Short-Term Memory)**
  - **GRU (Gated Recurrent Unit)**
- Cross-entropy loss funkcija  
- AdamW optimizator  
- Dropout i gradient clipping  

Implementacija je izrađena korištenjem **PyTorch** biblioteke.

---

## Instalacija
```bash
pip install torch numpy
````

Preporučeno je koristiti **Python 3.10+**.
Ako je dostupna CUDA podrška, PyTorch će automatski koristiti GPU.

---

## Predobrada podataka

Iz sirovog korpusa (`corpus.txt`) generira se dataset pogodan za treniranje:

```bash
python src/data_prep.py --input data/raw --output data/processed --seq-len 128
```

Rezultat predobrade su:

* `dataset.npz` – tekst kodiran kao niz indeksa znakova
* `vocab.json` – mapiranje znak ↔ indeks

---

## Treniranje modela

### LSTM

```bash
python src/train.py --cell lstm --epochs 10
```

### GRU

```bash
python src/train.py --cell gru --epochs 10
```

Trenirani modeli spremaju se u direktorij:

```
models/
```

---

## Generiranje teksta

Nakon treniranja moguće je generirati nove tekstove pjesama:

```bash
python src/generate.py \
  --checkpoint models/lstm_last.pt \
  --prompt "love is " \
  --steps 500 \
  --temperature 1.1 \
  --top-k 50
```

Generirani tekst sprema se u direktorij `outputs/`.

---

## Evaluacija

Kvaliteta modela mjeri se pomoću **perpleksnosti**, koja opisuje koliko dobro model predviđa sljedeći znak u sekvenci:

```bash
python src/eval.py --checkpoint models/lstm_last.pt
python src/eval.py --checkpoint models/gru_last.pt
```

Niža vrijednost perpleksnosti označava bolji model.

---

## Rezultati i zaključak

Rezultati pokazuju da oba modela (LSTM i GRU) mogu generirati koherentne i strukturirane tekstove pjesama. GRU model postiže usporedive rezultate s LSTM-om uz manji broj parametara i bržu konvergenciju, dok LSTM pokazuje stabilnije učenje na duljim sekvencama.

Projekt uspješno demonstrira primjenu rekurentnih neuronskih mreža u kreativnom području obrade prirodnog jezika.

---

## Autor

Projekt izrađen u sklopu kolegija **Neuronske mreže**.

