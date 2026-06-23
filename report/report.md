# Scientific report

## 1. Streszczenie artykułu

Vaswani et al. (2017) proponują Transformer, czyli model sekwencja-do-sekwencji oparty wyłącznie na mechanizmie uwagi, bez rekurencji ani splotów. Na tłumaczeniu maszynowym EN -> DE (zbiór WMT 2014) Transformer osiąga 27,3 punktu BLEU w wariancie bazowym i 28,4 w dużym, bijąc wszystkie wcześniejsze modele, w tym ensemble'e, o ponad 2 punkty BLEU. Co ważne, robi to przy niższym koszcie obliczeniowym: wariant duży kosztuje 2,3 × 10¹⁹ FLOP, podczas gdy najlepszy ensemble (GNMT+RL) potrzebował 1,8 * 10^20 FLOP, czyli niemal osiem razy więcej (Tabela 2). Autorzy formułują też drugie twierdzenie, weryfikowane w Tabeli 3: multi-head attention jest lepsze od single-head przy tym samym budżecie parametrów, choć zbyt wiele heads zaczyna pogarszać wyniki.

Artykuł wprowadza trzy kluczowe nowości. Po pierwsze, attention zamiast rekurencji: w LSTM informacja płynie przez ukryte stany krok po kroku, więc dwie odległe pozycje sekwencji komunikują się przez wiele pośrednich kroków. W self-attention każda pozycja może bezpośrednio spojrzeć na każdą inną już w jednym kroku, co umożliwia pełną równoległość obliczeń. Po drugie, multi-head attention: zamiast obliczać jedną funkcję uwagi, model oblicza ich h równocześnie, każda w mniejszej podprzestrzeni, co pozwala modelowi jednocześnie śledzić różne typy zależności w sekwencji. Po trzecie, sinusoidalne kodowanie pozycji: ponieważ self-attention nie rozróżnia kolejności tokenów, autorzy dodają do embeddingów stały wzorzec sinusoid niosący informację o pozycji, bez żadnych uczonych parametrów.

Transformer stał się fundamentem praktycznie każdego nowoczesnego dużego modelu językowego: BERT, GPT-2/3/4 i T5 to jego bezpośrednie pochodne.

Oryginalny eksperyment tłumaczeniowy jest nieosiągalny na laptopie: model bazowy ma 65M parametrów, był trenowany przez 12 godzin na 8 GPU P100, a zbiór danych to 4,5 miliona par zdań. Zamiast tego odtwarzamy ducha Tabeli 3 z artykułu: sprawdzamy, czy wielogłowicowa uwaga jest lepsza od jednogłowicowej przy tej samej liczbie parametrów. Robimy to na syntetycznym zadaniu odwracania sekwencji i porównujemy Transformery z h ∈ {1, 2, 4, 8} heads z modelem LSTM jako punktem odniesienia.

---

## 2. Metoda

Poniżej opisujemy kluczowe komponenty Transformera własnymi słowami, zgodnie z notacją z Vaswani et al. (2017). 

### 2.1 Skalowana uwaga iloczynowa

Podstawowy blok to *scaled dot-product attention*. Mamy trzy macierze: zapytania **Q**, klucze **K** i wartości **V**. Wyjście liczymy wzorem:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V                   
```

Iloczyn QKᵀ mówi, jak bardzo każde zapytanie pasuje do każdego klucza. Softmax zamienia te wyniki w wagi, a następnie bierzemy ważoną sumę wartości V. Dzielenie przez √d_k zapobiega sytuacji, w której przy dużych wymiarach iloczyny skalarne stają się tak duże, że gradient softmaxa zanika. Autorzy uzasadniają to w przypisie 4: jeśli składowe q i k mają wariancję 1, to q·k ma wariancję d_k, więc skalowanie przywraca wariancję do 1.

W dekoderze dodajemy maskę przyczynową, żeby token na pozycji i nie widział tokenów z pozycji j > i:

```
score(i, j) = -∞   jeśli j > i
```

W implementacji jest to górnotrójkątna macierz logiczna z `torch.triu(..., diagonal=1)` przekazywana jako argument `tgt_mask`.

### 2.2 Multi-head attention 

Zamiast jednej funkcji uwagi model liczy ich h, każda na wektorach o wymiarze d_k = d_model/h:

```
head_i = Attention( Q W_i^Q,  K W_i^K,  V W_i^V )                 

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O              
```

Kluczowa właściwość: przy stałym d_model całkowita liczba parametrów w projekcjach nie zależy od h. Wynika to stąd, że h heads wymiaru d_model/h ma łącznie tyle samo parametrów co jeden head wymiaru d_model. Dzięki temu ablacja liczby heads jest uczciwa: modele różnią się podziałem przestrzeni reprezentacji, a nie pojemnością. Każdy head może wyspecjalizować się w innym typie zależności, a projekcja W^O scala wyniki z powrotem do wymiaru d_model.

### 2.3 Kodowanie pozycyjne 

Self-attention traktuje wejście jak nieuporządkowany zbiór: jeśli przestawimy tokeny, wyjście przestawi się tak samo. Żeby model wiedział o kolejności, autorzy dodają do embeddingów stały wektor pozycyjny przed pierwszą warstwą:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )                
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )                
```

Każda pozycja otrzymuje unikalny wzorzec sinusoid o geometrycznie rozłożonych częstotliwościach. Kodowanie jest deterministyczne, obliczone z góry i nie wymaga uczenia. W naszej implementacji embedding jest najpierw mnożony przez √d_model (zgodnie z zaleceniem autorów), a dopiero potem dodajemy kodowanie pozycyjne, żeby oba miały zbliżoną skalę.

### 2.4 Sieć feed-forward 

Po bloku uwagi każda warstwa zawiera prostą dwuwarstwową sieć w pełni połączoną, stosowaną niezależnie do każdej pozycji:

```
FFN(x) = max(0, x W₁ + b₁) W₂ + b₂                            
```

W oryginalnym artykule d_ff = 2048, czyli czterokrotność d_model = 512. U nas d_ff = 256, co daje podobną proporcję względem d_model = 128.

### 2.5 Połączenia rezydualne i normalizacja warstwy

Każdy podblok (uwaga lub FFN) jest opakowany w połączenie rezydualne z normalizacją warstwy:

```
Wyjście = LayerNorm( x + Podblok(x) )                            
```

Połączenie rezydualne ułatwia przepływ gradientów przez głęboką sieć. LayerNorm normalizuje aktywacje wzdłuż wymiaru cech niezależnie dla każdej pozycji, co stabilizuje trening bez uzależnienia od rozmiaru batcha. Używamy `norm_first=False`, czyli post-norm, zgodnie z wersją z artykułu.

### 2.6 Architektura enkoder–dekoder 

Model składa się z enkodera i dekodera, każdy z N warstw.

Enkoder przetwarza sekwencję źródłową: każda warstwa to self-attention z maską paddingu, po którym następuje FFN. Wyjście enkodera to ciąg wektorów nazywany w kodzie *memory*.

Dekoder generuje sekwencję docelową token po tokenie. Każda warstwa zawiera trzy podbloki: maskowaną self-attention (maska przyczynowa zapobiega podglądaniu przyszłości), cross-attention do pamięci enkodera, oraz FFN.

Podczas treningu stosujemy *teacher forcing*: dekoder otrzymuje na wejście prawdziwe tokeny docelowe. Podczas ewaluacji generujemy zachłannie, krok po kroku, podając własne poprzednie wyjście.

### 2.7 Baseline LSTM

Jako punkt porównawczy trenujemy enkoder–dekoder LSTM. Enkoder czyta całą sekwencję i produkuje końcowy stan ukryty, którym inicjalizujemy dekoder. Nie ma tu żadnego mechanizmu uwagi. Szerokość ukryta wynosi d_model = 128, przez co LSTM ma około 540K parametrów wobec 675K Transformera. Różnica wynika z oddzielnych tablic embeddingów dla źródła i celu w Transformerze.

### 2.8 Konfiguracje modeli dla ablacji

Wszystkie cztery warianty Transformera mają identyczne hiperparametry: d_model = 128, d_ff = 256, N = 2 warstwy, dropout = 0,1. Różni je wyłącznie liczba heads:

| h (heads) | d_k = d_model/h | Liczba parametrów |
|-------------|-----------------|-------------------|
| 1           | 128             | ≈ 675 K           |
| 2           | 64              | ≈ 675 K           |
| 4           | 32              | ≈ 675 K           |
| 8           | 16              | ≈ 675 K           |

Stałość liczby parametrów jest weryfikowana testem jednostkowym `test_parameter_counts_identical_across_head_configs` w `tests/test_architecture.py`.

---

## 3. Experimental setup

We use a deterministic synthetic sequence-reversal task. Input tokens are sampled
uniformly from a vocabulary of 29 data symbols; token 0 is reserved for padding,
token 1 begins decoder input (`BOS`), and token 2 is an explicit end-of-sequence
marker (`EOS`). The marker makes the
length observable when examples of different lengths share a padded batch. The
training set contains 10,000 sequences, while the validation and in-distribution
test sets contain 1,000 sequences each. Their lengths are sampled uniformly from
5 to 20 tokens. A separate 1,000-example test set contains lengths 21–40 and is
used only to measure out-of-distribution length generalization. The four splits
are generated from independent, recorded pseudorandom seeds.

The primary model is a two-layer encoder–decoder Transformer with sinusoidal
positional encoding, model width 128, feed-forward width 256, ReLU activation,
and dropout 0.1. A two-layer encoder–decoder LSTM with the same embedding and
hidden width is used as a baseline. Both models use teacher forcing during
training and greedy autoregressive decoding during evaluation. For the ablation,
the Transformer uses 1, 2, 4, or 8 heads
while keeping `d_model=128`; therefore the projection parameter count is fixed
and only the per-head width changes.

Models are trained with AdamW, learning rate 0.001, weight decay 0.0001, batch
size 64, cross-entropy loss ignoring padding, and gradient norm clipping at 1.0.
Training lasts at most 20 epochs and stops after five epochs without improvement
in teacher-forced validation loss. Autoregressive decoding is deliberately
reserved for the final validation and test passes to keep the sweep feasible.
Final configurations are run with seeds 13, 37, and 71. We report token accuracy, exact-sequence accuracy,
trainable parameter count, and wall-clock training time. Raw per-epoch histories,
checkpoints, configuration, software versions, and hardware metadata are saved
under `artifacts/`.

The final sweep was executed on Windows 11 using Python 3.12.13 and CPU-only
PyTorch 2.12.1 on 16 logical CPU threads. Individual LSTM runs took about
5.7 minutes on average, while Transformer runs took approximately 10.5–12.1
minutes on average depending on head count; the longest run remained well below
the four-hour hardware constraint.

These settings are a deliberate laptop-scale deviation from Vaswani et al.
(2017): the original WMT translation datasets and base/large models are replaced
by a controlled algorithmic task and a tiny model. The head-count ablation keeps
the central comparison of differently partitioned attention representations,
but its absolute results should not be interpreted as translation quality.

During protocol development we piloted a simpler encoder-only token classifier.
It plateaued at approximately 23% token accuracy and 0% exact-sequence accuracy,
even after adding an explicit `EOS` token. We rejected that simplification before
the multi-seed sweep and adopted the encoder–decoder protocol above.

## 4. Results

## 5. Ablation: number of attention heads

## 6. Limitations and reproducibility notes

## 7. Conclusions

## References
