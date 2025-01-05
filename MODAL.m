function [frequency_sliding,bands,bandpow,bandphases] = MODAL(signal,params)
% 多重振荡检测算法 (MOD-AL)
% 该算法用于计算（神经）信号的瞬时功率、相位和频率，
% 并在功率超出信号“背景”全局1/f拟合（可选：局部1/f拟合）时，自适应地识别频带。
% 作者：Andrew J Watrous，2017年10月
% 瞬时频率估计采用“频率滑动”（Frequency Sliding）方法
% 详细说明见：http://mikexcohen.com/data/Cohen2014_freqslide.pdf

% 输入：
% signal - 待分析的信号，可以是任何神经时间序列数据。
% params 必须包括以下参数：
%   srate - 信号的采样率（单位：Hz）。
%   wavefreqs - 用于背景拟合的频率范围，建议最大频率至少为30Hz，以保证1/f拟合的良好效果。
% 可选参数：
% bad_data: 布尔向量，标识坏数据的位置。1表示需要排除的坏数据，长度必须与信号一致。
% local_winsize_sec（默认值为10秒）：用于局部拟合的不同时间窗大小的向量
% （例如：[1 5 10]，单位为秒）。如果为空，则不计算局部拟合/阈值，
% 并将返回所有时间点的功率、相位和频率估计。
% params.crop_fs = 布尔值，是否裁剪超出频带的频率估计。
% wavecycles - 小波周期数（默认为6）。

%f0 = 10; % 正弦波频率为 10 Hz
%t = 0:1/1000:1-1/1000; % 时间向量，采样率 1000 Hz，时长 1 秒
%signal = sin(2 * pi * f0 * t); % 10 Hz 的正弦波 （1000，1）
%params.wavefreqs = 1:0.5:30; % 1 到 30 Hz 的频率范围
%params.srate = 1000; % 采样率为 1000 Hz


% 输出：
% frequency sliding - 信号在每个频带内的瞬时频率（矩阵大小：频带数 x 样本数）。
% bands - 每个检测频带的边界（矩阵大小：频带数 x 2，表示频带的上下界）。
% bandpow - 信号在每个检测频带内的平均功率（矩阵大小：频带数 x 样本数）。
% bandphase - 信号在每个检测频带内的瞬时相位（矩阵大小：频带数 x 样本数）。
% 核心步骤：
% 1. 自适应识别超过背景1/f的窄带振荡，
%    方法参考Lega, Jacobs, & Kahana（2012年，Hippocampus期刊）。
% 2. 在每个频带中计算“频率滑动”（参考MX Cohen 2014年，JNeuroscience期刊）。
%    注意，这些频率估计是频带内的连续值，并未舍入到具体的wavefreqs。
% 3. 默认情况下，移除功率、相位和频率低于局部1/f拟合线的估计值。
% 4. 移除由于相位滑移引起的超出检测频带的频率估计，
%    详细说明参考Cohen 2014年，JNeurosci期刊，图1B及其说明。
% 其他说明：
% 需要使用Kahana实验室的eegtoolbox中的“multiphasevec2”函数，
% 或者替换为你自己的函数，用于计算频率x时间的功率矩阵。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% 开始代码
% 初始化输入参数
wavefreqs = params.wavefreqs; % 提取用于分析的频率范围
if isfield(params, 'local_winsize_sec') 
    wins = params.local_winsize_sec * params.srate; % 如果提供了局部窗口大小（以秒为单位），计算对应的采样点数
else
    wins = params.srate .* 10; % 默认情况下，单窗口长度为10秒
end
if isfield(params, 'wavecycles')
    wavecycles = params.wavecycles; % 如果提供了小波周期数，则使用提供的值
else
    wavecycles = 6; % 默认设置为6个小波周期
end
if isfield(params, 'crop_fs') % 是否裁剪频率估计，默认为裁剪
    crop_fs = params.crop_fs;
else
    crop_fs = 1; % 默认值为1（裁剪）
end
% 如果信号的样本点在第二维度，将其转置为第一维度（确保样本点是第一维度）
if size(signal, 2) > size(signal, 1)
    signal = signal'; 
end
% 确保信号是以均值为中心的（均值中心化），
% 这样希尔伯特变换以及功率/相位估计才有效
signal = signal - nanmean(signal); %（1000，1）
% 自适应部分
% 使用 Kahana 的 eegtoolbox 提供的函数
% http://memory.psych.upenn.edu/files/software/eeg_toolbox/eeg_toolbox.zip
% 使用小波提取频率 x 时间矩阵的功率估计值
[~, pow] = multiphasevec2(params.wavefreqs, signal', params.srate, wavecycles); %(59,1000) 59：wavefreqs个数 pow：信号在某个频率和时间点上的功率大小
% 处理坏数据：将坏时间段的功率值替换为 NaN。
% 在频带识别时会排除这些数据
if isfield(params, 'bad_data')
    bad_idx = find(params.bad_data == 1); % 找到坏数据的索引
    pow(:, bad_idx) = NaN; % 对应位置的功率值设为 NaN
end
% 核心步骤1：对1/f进行拟合以自适应地识别频带
[bands, bandidx, bandpow] = GetBands(wavefreqs, pow);
% freq_bands: 每个频带的频率范围，大小为 [N, 2] 的矩阵  [9.5, 10.5] Hz ，N为识别到几个
% bandidx: 每个频带对应的 wavefreqs 索引，存储为 cell 数组 [18 19 20] （wavefreqs(19) = 10）
% bandpow: 每个频带在时间上的平均功率值，大小为 [N, length(signal)] 

% 核心步骤 #2：基于 MX Cohen 的频率滑动代码
%% 滤波数据
% 应用带通滤波器，带有 15% 的过渡区域
FS = zeros(size(bands,1),length(signal)).*NaN; % 初始化 FS（即频带内的所有频率滑动估计）
bandphases = zeros(size(bands,1),length(signal)).*NaN; % 初始化相位
for iBand = 1:size(bands,1) % 遍历每个频带
    freq_bands = bands(iBand,:); % 获取当前频带的上下边界
    trans_width = .15; % 设置过渡区域宽度为 15%
    idealresponse = [ 0 0 1 1 0 0 ]; % 理想滤波器响应
    filtfreqbounds = [ 0 (1-trans_width)*freq_bands(1) freq_bands(1) freq_bands(2) freq_bands(2)*(1+trans_width) params.srate/2 ]/(params.srate/2); % 滤波频率边界
    filt_order = round(2*(params.srate/freq_bands(1))); % 计算滤波器阶数
    filterweights = firls(filt_order, filtfreqbounds, idealresponse); % 构建滤波器
    filtered_signal = filtfilt(filterweights, 1, signal); % 使用滤波器对信号进行双向滤波
    
    % 对滤波后的信号进行 Hilbert 变换
    temphilbert = hilbert(filtered_signal); % 计算 Hilbert 变换
    anglehilbert = angle(temphilbert); % 提取瞬时相位
    bandphases(iBand,:) = anglehilbert; % 存储频带内的相位
    
    % 来自 MX Cohen 的代码
    frompaper = params.srate * diff(unwrap(anglehilbert)) / (2 * pi); % 根据 Cohen 2014 论文计算瞬时频率
    frompaper(end+1) = NaN; % 处理差分导致丢失一个样本点的问题
    time_wins = [.05 .2 .4]; % 来自 Cohen 的时间窗口（以秒为单位的分数）
    orders = time_wins * params.srate; % 根据时间窗口计算滤波顺序
    % 将信号分为 10 段以提高处理效率
    % 注意：即使使用 parfor，也不会明显提高速度
    numchunks = 10; % 定义分段数
    chunks = floor(linspace(1, length(frompaper), numchunks)); % 划分为等间隔的时间段
    
    meds = zeros(length(orders), length(frompaper)); % 初始化存储不同窗口的中值滤波结果
    for iWin = 1:length(orders) % 使用不同窗口大小进行中值滤波
        for iChunk = 2:numchunks
            chunkidx = chunks(iChunk-1):chunks(iChunk)-1; % 避免双重计算边界，最后一个样本会被排除
            meds(iWin, chunkidx) = medfilt1(frompaper(chunkidx), round(orders(iWin))); % 对分段数据应用中值滤波
        end
    end
    
    % 对不同窗口的中值结果取中值
    median_of_meds = median(meds);
    
   % 核心步骤 #4：将频带范围外的频率估计值设为 NaN
    clear below* above* outside* % 清除中间变量
    if crop_fs
        below_idx = (median_of_meds < bands(iBand,1)); % 找到低于频带下边界的索引
        above_idx = (median_of_meds > bands(iBand,2)); % 找到高于频带上边界的索引
        outside_idx = find([below_idx + above_idx] == 1); % 识别频带范围外的索引
        median_of_meds(outside_idx) = NaN; % 将频带范围外的频率估计值设为 NaN
    end
    FS(iBand,:) = median_of_meds; % 存储频带内的所有频率滑动估计
end
% 可选的核心步骤 #3：  
% 在较小的时间窗口内进行 1/f 拟合，并将低于 1/f 拟合的功率、相位和频率估计替换为 NaN
if size(bands,1)>0 % 检查是否检测到频带
    frequency_sliding = FS;zeros([size(FS) length(wins)]); % 初始化频率滑动变量
    for iW = 1:length(wins) % 遍历不同的局部拟合窗口大小
        winsize = wins(iW); % 获取当前窗口大小
        for iWin = 1:winsize:length(signal) % 遍历信号，按照窗口大小划分
            windex = iWin:iWin+winsize; % 定义当前窗口内的样本索引
            if windex(end)>length(signal) % 如果窗口超过信号长度
                windex = iWin:length(signal); % 调整窗口至信号末尾
            end
            % 添加 if 语句以处理由于 bad_data 导致窗口内所有值均为 NaN 的情况（2023 年 11 月 1 日更新）
            if sum(sum(isnan(pow(:,windex)))) < (length(windex) .* length(wavefreqs)) % 如果窗口内并非全为 NaN
                % 输入所有频率滑动估计，输出修剪后的版本
                [frequency_sliding(:,windex,iW)] = ...
                fit_one_over_f_windows(FS(:,windex), wavefreqs, pow(:,windex), bandidx);
            else
                % 如果窗口内全为 NaN，将该窗口的频率滑动值设置为 NaN
                frequency_sliding(:,windex,iW) = NaN;
            end
        end
    end
    
    % 对不同窗口大小的结果取 NaN 平均值，从而在多个窗口大小的情况下，将超过 1/f 拟合的点保留
    frequency_sliding = nanmean(frequency_sliding, 3); 
else % 如果没有检测到任何频带
    frequency_sliding = NaN;
end
% 将频率估计为 NaN 的位置在相位和带功率中也设置为 NaN
% 确保在所有度量中具有一致的估计点
bandpow(isnan(frequency_sliding)) = NaN;
bandphases(isnan(frequency_sliding)) = NaN;
% 将所有数据类型转换为 single 类型以节省空间
bandpow = single(bandpow);
bandphases = single(bandphases);
frequency_sliding = single(frequency_sliding);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%subfunctions below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%从 1/f 拟合中识别频带
function  [freq_bands, bandidx, bandpow] = GetBands(wavefreqs, pow)
% 输入：wavefreqs 和 pow 未进行对数变换
% 输出：
% freq_bands: 频带的上下边界（单位：Hz）
% bandidx: 每个频带中 wavefreqs 的索引
% bandpow: 每个频带的平均对数功率
fz = log(wavefreqs); % 对频率进行对数变换
mean_pow = log(nanmean(pow, 2)); % 使用 nanmean 计算平均功率，以处理 bad_data 导致的 NaN 值
[b, ~] = robustfit(fz, mean_pow); % 核心拟合步骤：使用 robustfit 进行 1/f 拟合
fit_line = b(1) + b(2) .* fz; % 计算拟合的 1/f 线
above1f = (mean_pow - fit_line') > 0; % 找到功率高于 1/f 拟合线的点
bw = bwlabel(above1f); % 标记连续的频带
ctr = 1; % 频带计数器
for iBand = 1:max(unique(bw)) % 遍历所有检测到的频带
    idx = find(bw == iBand); % 找到当前频带的索引
    if length(idx) > 1 % 确保频带包含多个点，而不是单个频率
        freq_bands(ctr, 1) = wavefreqs(min(idx)); % 频带的下边界
        freq_bands(ctr, 2) = wavefreqs(max(idx)); % 频带的上边界
        bandidx{ctr} = idx; % 存储频带索引
        bandpow(ctr, :) = log(mean(pow(idx, :))); % 计算频带的平均对数功率
        crit_pow = mean(fit_line(idx)); % 计算频带的关键功率（可选）
        ctr = ctr + 1; % 增加频带计数
    end
end
% 结束频带识别子函数
% 核心步骤 #3：移除低于局部 1/f 拟合的频率滑动估计
function [frequency_sliding] = fit_one_over_f_windows(frequency_sliding, wavefreqs, pow, bandidx)
% 输入：所有频带内的频率滑动估计
% 输出：高于局部 1/f 拟合的频率滑动估计
% 与其他子函数相同的拟合过程
fz = log(wavefreqs);
local_mean_pow = log(nanmean(pow, 2)); % 使用 nanmean 计算局部平均功率，以处理 bad_data 导致的 NaN 值
[b, ~] = robustfit(fz, local_mean_pow); % 局部 1/f 拟合
local_fit_line = b(1) + b(2) .* fz; % 计算局部拟合线
% 检查每个时间点上的频率估计是否高于局部 1/f 拟合线
logpow = log(pow);
fitpow = repmat(local_fit_line, size(logpow, 2), 1)'; % 将局部拟合线扩展到与 logpow 相同的维度
powdiff = logpow - fitpow; % 计算功率差值
threshpow = (powdiff > 0); % 找到功率高于拟合线的点
tmpfs = frequency_sliding;
for iB = 1:length(bandidx) % 遍历所有频带
    % 找到频率滑动估计非 NaN 的时间点
    idx1 = [];
    idx1 = find(~isnan(frequency_sliding(iB, :)) == 1); % 找到非 NaN 的 FS 估计
    if ~isempty(idx1) % 如果存在非 NaN 的 FS 估计
        fswf = [];
        for iT = 1:length(idx1) % 遍历非 NaN 时间点
            fswf(iT) = dsearchn(wavefreqs', frequency_sliding(iB, idx1(iT))); % 将频率滑动估计值映射到 wavefreqs 索引
        end
        subz = sub2ind(size(threshpow), fswf, idx1); % 获取二维矩阵中的索引
        threshvalz = [];
        threshvalz = threshpow(subz); % 检查这些点是否高于阈值
        tmpfs(iB, idx1(find(threshvalz == 0))) = NaN; % 将低于阈值的 FS 估计替换为 NaN
    else
        tmpfs(iB, :) = NaN; % 如果没有非 NaN 的 FS 估计，将整个频带的 FS 估计设置为 NaN
    end
end
frequency_sliding = tmpfs; % 返回修正后的频率滑动估计
