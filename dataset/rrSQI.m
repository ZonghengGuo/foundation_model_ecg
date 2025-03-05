function r = rrSQI(ECG,qrs,freq)
% rrSQI  RR signal quality index.
%   [BeatQ,BeatN, r] = rrSQI(ECG,qrs) returns a binary signal quality
%   assessment of each RR.
%   In:   ECG    <mx1> --- ECG waveform
%         qrs    <nx1> --- qrs R peak detection by using eplim.m (nicolò)
%
%   Out:  BEATQ    <nx10> --- SQI of each beat: 0=good, 1=bad
%           Col 1:  logical value        (col2 OR col3 OR col4)
%               2:  HR  not physiologic  (<20 or >200 bpm)
%               3:  abnormal change      (beat-to-beat change > 30% of preceeding beat)
%               4:  abnormal period      (beat-to-beat change > 0.5 sec)
%               5:  noisy beat           (detected on ECG waveform)
%               6:  noisy ECG            (random noise on ECG)
%         BEATN    <nX1> --- SQI of ECG corresponding beat
%                   logical value (col2 OR col3 OR col4)
%
%         R        <1x1>  fraction of good beats in RR
% manuela.ferrario@polimi.it 9/10/2014

if (length(qrs)<20 & length(ECG)<200)
    BeatQ = [];
    r = [];
    return
end


fs=freq;
timeECG=(0:length(ECG)-1)'./fs;
RR=diff(timeECG(qrs));


%treshold

rangeHR     = [40 120]; % bpm
dHR         = .30;
dPeriod     = 0.5;       % 64 samples = 0.5 second
noiseEN     = 2;

% beat quality
HR=60./RR;
badHR       = find(HR    < rangeHR(1)  |     HR > rangeHR(2));

jerkPeriod  = 1+ find(abs(diff(RR)) > dPeriod);
jerkHR      = 1+ find(abs(diff(HR)./HR(1:end-1)) > dHR);


% ecg quality
w=fs*1; %1 sec window
E=[];k=0;
ecg=detrend(ECG)./std(ECG)+10;

for i=1:w:length(ECG)-w
    k=k+1;
    e=ecg(i:i+w);
    E(k,1)=sum(e);
    [apen(k), sampen(k)]=apsampen(e,1,0.1,0);
    I(k)=i;
end

B=ceil(qrs/w);
B=B(1:end-1);
j=find(B>length(E));
if ~isempty(j)
    B(j)=length(E);
end
    
noise=[E(B) sampen(B)'];

M=prctile(E,95);
j=find(noise(:,1)>M);
jj=find(noise(:,2)>noiseEN);


bq=zeros(length(qrs)-1,6);
bq(badHR,            2) = 1;
bq(jerkPeriod,       3) = 1;
bq(jerkHR,           4) = 1;
bq(j,                5) = 1;
bq(jj,               6) = 1;

bq(:,1) = bq(:,2)|bq(:,3)|bq(:,4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make all "...101..." into "...111..."
y = bq(:,1);
y(find(diff(y,2)==2)+1)=1;
bq(:,1)=y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BeatQ   = logical(bq);

% fraction of good beats overall
r = length(find(bq(:,1)==0))/length(qrs);

bn = bq(:,5)|bq(:,6);

BeatN   = logical(bn);


