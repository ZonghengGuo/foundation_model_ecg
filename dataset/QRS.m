function r = QRS(ecg,fs)

mex eplim.cpp QRSFILT.cpp

fc_ecg = 200;
% Ricampioniamo ECG a 200Hz e applichiamo EP Lim  (200 Hz LO RICHIEDE EPlim, magari chiedere conferma a riccardo)
% come prima sta estraendo il segnale ecg e ricampionando
tx = 0:1/fs:length(ecg)/fs-1/fs; 
ecg = resample(ecg,tx,fc_ecg)';

% applico un filtro passa-banda
[b,a] = butter(4,2*[8 20]/fc_ecg,'bandpass'); % Creazione di un filtro passa-banda
ecg = filtfilt(b,a,ecg); % Applicazione del filtro
                         % un filtro Butterworth passa-banda di ordine 4 con una banda passante tra 8 e 20 Hz 
                         % (frequenze comunemente associate ai componenti principali dell'ECG) e una frequenza di campionamento fc_ecg

% Orlo vettore per evitare di perdere campioni -> "Orlare" il vettore probabilmente significa estendere il vettore ecg 
                                                 % per evitare problemi ai bordi (agli estremi del segnale) durante operazioni che potrebbero richiedere 
                                                 % un contesto più ampio, vale a dire per fornire più campioni all'inizio e evitare distorsioni
                                                 % Di solito lo si fa pre-filtraggio perchè i bordi potrebbero subire distorsioni
ecg=[ecg(1:1500); ecg]; % aggiungo 1500 campioni del segnale stesso prima 

% identifico i complessi QRS con eplim
QRS=eplim(ecg*10000); % moltiplichiamo per controllare l'ampiezza 
                      % amplificare l'ampiezza del segnale ECG in alcuni casi facilita l'identificazione delle caratteristiche (i complessi QRS) 
                      % quando si utilizzano algoritmi che si basano su soglie fisse o valori numerici sensibili.

r=QRS(find(QRS)); % restituisce solo le righe con gli indici degli elementi non nulli nel vettore QRS
r=r-1500; % compensa l'orlo. Corregge le posizioni dei complessi QRS per riportarle alla loro posizione originale nel segnale ECG senza l'orlo.
ecg=ecg(1501:end); % elimina l'orlo da ecg

end