# P_01_Projekt

## Przedmiot: Analiza danych jakościowych i Text Mining

### Autor:

Dane pochodzą z 2 subredditów: [r/Amd](https://www.reddit.com/r/Amd/top/?t=year)
oraz [r/nvidia](https://www.reddit.com/r/nvidia/top/?t=year) zostały z nich
pobrane najpopularniejsze posty z ostatniego roku.
Dane zostały pobrane za pomocą modułu [praw](https://praw.readthedocs.io/en/stable/index.html)

W projekcie dane zostały już przetworzone i są dostępne w pliku [Raport.html](Raport.html)
jednakże jest także dostępna interaktywna wersja w pliku [Raport.ipynb](Raport.ipynb).
Dodatkowo wykresy i wyniki analiz zapisywane są w folderze [results](results)

W obu plikach raportu znajduje się tylko interpretacja, opisy i wyniki.
Dane przetwarzane są w odpowiednich modułach z katalogu [modules](modules)

Dla łatwiejszego uruchamiania stworzony został plik [main.py,](main.py) poprzez którego uruchomienie
można wykonać caly proces.
Od pobrania danych do wygenerowania wykresów

Jednakże do pobrania danych wymagane jest posiadanie konta na reddicie i utworzenie aplikacji
w celu uzyskania kluczy API. Klucze te należy wpisać w pliku konfiguracyjnym autoryzacji,
do którego następnie należy podać ścieżkę w pliku konfiguracyjnym [definitions](definitions.env)
W pliku konfiguracyjnym można także zmienić subreddity, z których pobieramy dane i inne parametry
Na domyślnych ustawieniach przetwarzanie danych z pobieraniem będzie trwało około 4 godzin
