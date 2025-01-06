function [frequency_sliding,bands,bandpow,bandphases] = MODAL(signal,params)
wavefreqs = params.wavefreqs;
if isfield(params, 'local_winsize_sec') 
    wins = params.local_winsize_sec * params.srate;
else
    wins = params.srate .* 10;
end
if isfield(params, 'wavecycles')
    wavecycles = params.wavecycles;
else
    wavecycles = 6;
end
if isfield(params, 'crop_fs')
    crop_fs = params.crop_fs;
else
    crop_fs = 1;
end
if size(signal, 2) > size(signal, 1)
    signal = signal'; 
end
signal = signal - nanmean(signal);
[~, pow] = multiphasevec2(params.wavefreqs, signal', params.srate, wavecycles);
if isfield(params, 'bad_data')
    bad_idx = find(params.bad_data == 1);
    pow(:, bad_idx) = NaN;
end
[bands, bandidx, bandpow] = GetBands(wavefreqs, pow);
FS = zeros(size(bands,1),length(signal)).*NaN;
bandphases = zeros(size(bands,1),length(signal)).*NaN;

for iBand = 1:size(bands,1)
    freq_bands = bands(iBand,:);
    trans_width = .15;
    idealresponse = [ 0 0 1 1 0 0 ];
    filtfreqbounds = [ 0 (1-trans_width)*freq_bands(1) freq_bands(1) freq_bands(2) freq_bands(2)*(1+trans_width) params.srate/2 ]/(params.srate/2);
    filt_order = round(2*(params.srate/freq_bands(1)));
    filterweights = firls(filt_order, filtfreqbounds, idealresponse);
    filtered_signal = filtfilt(filterweights, 1, signal);
    temphilbert = hilbert(filtered_signal);
    anglehilbert = angle(temphilbert);
    bandphases(iBand,:) = anglehilbert;
    frompaper = params.srate * diff(unwrap(anglehilbert)) / (2 * pi);
    frompaper(end+1) = NaN;
    time_wins = [.05 .2 .4];
    orders = time_wins * params.srate;
    numchunks = 10;
    chunks = floor(linspace(1, length(frompaper), numchunks));
    meds = zeros(length(orders), length(frompaper));
    for iWin = 1:length(orders)
        for iChunk = 2:numchunks
            chunkidx = chunks(iChunk-1):chunks(iChunk)-1;
            meds(iWin, chunkidx) = medfilt1(frompaper(chunkidx), round(orders(iWin)));
        end
    end
    median_of_meds = median(meds);
    clear below* above* outside*
    if crop_fs
        below_idx = (median_of_meds < bands(iBand,1));
        above_idx = (median_of_meds > bands(iBand,2));
        outside_idx = find([below_idx + above_idx] == 1);
        median_of_meds(outside_idx) = NaN;
    end
    FS(iBand,:) = median_of_meds;
end

if size(bands,1)>0
    frequency_sliding = FS;zeros([size(FS) length(wins)]);
    for iW = 1:length(wins)
        winsize = wins(iW);
        for iWin = 1:winsize:length(signal)
            windex = iWin:iWin+winsize;
            if windex(end)>length(signal)
                windex = iWin:length(signal);
            end
            if sum(sum(isnan(pow(:,windex)))) < (length(windex) .* length(wavefreqs))
                [frequency_sliding(:,windex,iW)] = ...
                fit_one_over_f_windows(FS(:,windex), wavefreqs, pow(:,windex), bandidx);
            else
                frequency_sliding(:,windex,iW) = NaN;
            end
        end
    end
    frequency_sliding = nanmean(frequency_sliding, 3);
else
    frequency_sliding = NaN;
end
bandpow(isnan(frequency_sliding)) = NaN;
bandphases(isnan(frequency_sliding)) = NaN;
bandpow = single(bandpow);
bandphases = single(bandphases);
frequency_sliding = single(frequency_sliding);

function [freq_bands, bandidx, bandpow] = GetBands(wavefreqs, pow)
fz = log(wavefreqs);
mean_pow = log(nanmean(pow, 2));
[b, ~] = robustfit(fz, mean_pow);
fit_line = b(1) + b(2) .* fz;
above1f = (mean_pow - fit_line') > 0;
bw = bwlabel(above1f);
ctr = 1;
for iBand = 1:max(unique(bw))
    idx = find(bw == iBand);
    if length(idx) > 1
        freq_bands(ctr, 1) = wavefreqs(min(idx));
        freq_bands(ctr, 2) = wavefreqs(max(idx));
        bandidx{ctr} = idx;
        bandpow(ctr, :) = log(mean(pow(idx, :)));
        crit_pow = mean(fit_line(idx));
        ctr = ctr + 1;
    end
end

function [frequency_sliding] = fit_one_over_f_windows(frequency_sliding, wavefreqs, pow, bandidx)
fz = log(wavefreqs);
local_mean_pow = log(nanmean(pow, 2));
[b, ~] = robustfit(fz, local_mean_pow);
local_fit_line = b(1) + b(2) .* fz;
logpow = log(pow);
fitpow = repmat(local_fit_line, size(logpow, 2), 1)';
powdiff = logpow - fitpow;
threshpow = (powdiff > 0);
tmpfs = frequency_sliding;
for iB = 1:length(bandidx)
    idx1 = find(~isnan(frequency_sliding(iB, :)) == 1);
    if ~isempty(idx1)
        fswf = [];
        for iT = 1:length(idx1)
            fswf(iT) = dsearchn(wavefreqs', frequency_sliding(iB, idx1(iT)));
        end
        subz = sub2ind(size(threshpow), fswf, idx1);
        threshvalz = threshpow(subz);
        tmpfs(iB, idx1(find(threshvalz == 0))) = NaN;
    else
        tmpfs(iB, :) = NaN;
    end
end
frequency_sliding = tmpfs;
