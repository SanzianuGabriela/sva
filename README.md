# Atenuarea zgomotului din imagini

##  Ce e zgomotul?
Zgomotul este un semnal aleator, care afectează informația utilă conținută
într-o imagine. El poate apare de-a lungul unui lanț de transmisiune, sau
prin codarea și decodarea imaginii, și reprezintă un element perturbator nedorit. De obicei se încearcă eliminarea lui prin diverse metode de filtrare.


##  Scopul proiectului:

Realizarea unui sistem de vedere artificială care să permită testarea algoritmilor de atenuare a zgomotului pe o baza de date cu imagini zgomotoase.







##  Algoritmii folosiți:

+ **Filtru Gaussian**

Această metodă utilizează un filtru gaussian pentru a estima zgomotul într-o imagine sau semnal. Filtrul gaussian aplică o medie ponderată a valorilor pixelilor în vecinătatea fiecărui pixel, reducând astfel diferențele bruste între pixeli. Acest tip de filtru este eficient în atenuarea zgomotului de tip aditiv, cum ar fi zgomotul gaussiano-aderent.

+ **Filtru Aritmetic**

Această metodă utilizează o medie aritmetică a valorilor pixelilor dintr-o vecinătate pentru a reduce zgomotul. Fiecare pixel este înlocuit cu media valorilor vecine, inclusiv propria sa valoare. Această tehnică este simplă și ușor de implementat, dar poate duce la pierderea detaliilor fine din imagine.

+ **Filtru Median**

Această metodă utilizează mediana valorilor pixelilor dintr-o vecinătate pentru a estima valoarea corectă a unui pixel și a reduce zgomotul. Valorile pixelilor sunt sortate în ordine crescătoare sau descrescătoare, iar valoarea mediană (valoarea centrală) este atribuită pixelului. Această tehnică este eficientă în eliminarea zgomotului impulsiv, cum ar fi zgomotul de impuls sau salt.

+ **Filtru Geometric**

Această metodă utilizează o medie geometrică a valorilor pixelilor dintr-o vecinătate pentru a reduce zgomotul. Acest tip de filtru calculează radicalul valorilor pixelilor în vecinătatea fiecărui pixel și atribuie valoarea pixelului rezultatul acestei medii geometrice. Această tehnică poate fi eficientă în atenuarea zgomotului de tip multiplicative.

+ **DNCNN** 

DNCNN (Deep Convolutional Neural Network) este o arhitectură de rețea neurală profundă folosită în domeniul prelucrării imaginilor, în special în ceea ce privește denoise-ul (eliminarea zgomotului) în imagini. 
DNCNN se bazează pe rețele convoluționale profunde, care sunt arhitecturi de rețele neurale ce utilizează straturi de convoluții pentru a extrage caracteristici din imagini. Acest model utilizează ideea de învățare reziduală (residual learning), în care se încearcă să se învețe diferența (reziduul) dintre imaginea zgomotoasă și imaginea curată. Prin urmare, rețeaua se antrenează să învețe să genereze un reziduu corectat care poate fi adăugat apoi la imaginea zgomotoasă inițială pentru a obține imaginea denoisată.


![Arhitectura DNCNN](DNCNN-arhitectura.jpg)
