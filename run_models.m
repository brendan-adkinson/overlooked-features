
clear all

data_path = '/Users/brendan/Dropbox/analyses/ranked_edges/data';


%%%%% LOAD DATASETS %%%%%%

%%%%% PNC %%%%%%

%%% PNC EF
dataset_pnc_ef = struct();
% Load mat file
load(fullfile(data_path, 'connectomes_pnc_1291_with_combo_info_thresh_fixed.mat'));
dataset_pnc_ef.mats = permute(connectomes, [2, 3, 1]);
% Load behavioral data
dataset_pnc_ef.behav_table = readtable(fullfile(data_path, 'new_pnc_pheno_imaging_zscore.csv'));
% Load behavioral data (those without imaging) for PCA calculation
dataset_pnc_ef.external_behav_table = readtable(fullfile(data_path, 'pnc_pheno_behavior_zscore.csv'));
% Define which phenotypes to predict from the behavioral file)
dataset_pnc_ef.phenotypes = {'lnb_tp', 'pcet_acc2', 'pcpt_t_tp'};
% Define which variable to covary for
% dataset_pnc_ef.covar_table = readtable(fullfile(data_path, 'new_pnc_id_age_only_imaging_only.csv'));
% dataset_pnc_ef.covars = {'age'};

%%% PNC LANG
dataset_pnc_lang = struct();
load(fullfile(data_path, 'connectomes_pnc_1291_with_combo_info_thresh_fixed.mat'));
dataset_pnc_lang.mats = permute(connectomes, [2, 3, 1]);
dataset_pnc_lang.behav_table = readtable(fullfile(data_path, 'new_pnc_pheno_imaging_zscore.csv'));
dataset_pnc_lang.external_behav_table = readtable(fullfile(data_path, 'pnc_pheno_behavior_zscore.csv'));
dataset_pnc_lang.phenotypes = {'pvrt_cr', 'wrat_cr_std'};
% dataset_pnc_lang.covar_table = readtable(fullfile(data_path, 'new_pnc_id_age_only_imaging_only.csv'));
% dataset_pnc_lang.covars = {'age'};


%%%%% HBN %%%%%%
%%% HBN EF
dataset_hbn_ef = struct();
load(fullfile(data_path, 'connectomes_hbn_1110_with_combo_info_thresh_fixed.mat'));
dataset_hbn_ef.mats = permute(connectomes, [2, 3, 1]);
dataset_hbn_ef.behav_table = readtable(fullfile(data_path, 'hbn_pheno_imaging_zscore.csv'));
dataset_hbn_ef.external_behav_table = readtable(fullfile(data_path, 'hbn_pheno_behavior_zscore.csv'));
dataset_hbn_ef.phenotypes = {'NIH_Card_Sort_Age_Corr_Stnd', 'NIH_Flanker_Age_Corr_Stnd', 'NIH_List_Sort_Age_Corr_Stnd', 'NIH_Processing_Age_Corr_Stnd'};
% dataset_hbn_ef.covar_table = readtable(fullfile(data_path, 'hbn_id_sex_only_imaging_only.csv'));
% dataset_hbn_ef.covars = {'M'};

%%% HBN LANG
dataset_hbn_lang = struct();
load(fullfile(data_path, 'connectomes_hbn_1110_with_combo_info_thresh_fixed.mat'));
dataset_hbn_lang.mats = permute(connectomes, [2, 3, 1]);
dataset_hbn_lang.behav_table = readtable(fullfile(data_path, 'hbn_pheno_imaging_zscore.csv'));
dataset_hbn_lang.external_behav_table = readtable(fullfile(data_path, 'hbn_pheno_behavior_zscore.csv'));
dataset_hbn_lang.phenotypes = {'CTOPP_BW_S', 'CTOPP_EL_S', 'CTOPP_NR_S', 'CTOPP_RD_S', 'CTOPP_RL_S', 'TOWRE_PDE_Scaled', 'TOWRE_SWE_Scaled', 'TOWRE_Total_Scaled'};
% dataset_hbn_lang.covar_table = readtable(fullfile(data_path, 'hbn_id_sex_only_imaging_only.csv'));
% dataset_hbn_lang.covars = {'M'};


% %%%%% HCPD %%%%%%

%%% HCPD EF
dataset_hcpd_ef = struct();
load(fullfile(data_path, 'connectomes_hcpd_completecase_428.mat'));
dataset_hcpd_ef.mats = permute(connectomes, [2, 3, 1]);
dataset_hcpd_ef.behav_table = readtable(fullfile(data_path, 'hcpd_pheno_imaging_zscore.csv'));
dataset_hcpd_ef.phenotypes = {'list_sorting_age_corrected_standard_score', 'picseq_ageadjusted', 'cardsort_ageadjusted', 'flanker_ageadjusted', 'patterncomp_ageadjusted'};
% dataset_hcpd_ef.covar_table = readtable(fullfile(data_path, 'hcpd_id_age_only_imaging_only.csv'));
% dataset_hcpd_ef.covars = {'age'};


%%% HCPD LANG
dataset_hcpd_lang = struct();
load(fullfile(data_path, 'connectomes_hcpd_completecase_428.mat'));
dataset_hcpd_lang.mats = permute(connectomes, [2, 3, 1]);
dataset_hcpd_lang.behav_table = readtable(fullfile(data_path, 'hcpd_pheno_imaging_zscore.csv'));
dataset_hcpd_lang.phenotypes = {'readingtest_age_corrected_std', 'picturevocab_age_corrected_std'};
% dataset_hcpd_lang.covar_table = readtable(fullfile(data_path, 'hcpd_id_age_only_imaging_only.csv'));
% dataset_hcpd_lang.covars = {'age'};




%%%%% RUN PREDICTIONS %%%%%%

% Specify which datasets to run
dataset_dictionary = containers.Map({'hbn_ef', 'pnc_ef', 'hcpd_ef', 'hbn_lang', 'pnc_lang', 'hcpd_lang'}, {dataset_hbn_ef, dataset_pnc_ef, dataset_hcpd_ef, dataset_hbn_lang, dataset_pnc_lang, dataset_hcpd_lang});
dataset_names_all = keys(dataset_dictionary);

for my_seed = 1:100
    fprintf('Current seed is %d\n', my_seed);
    for ranked_segment = 1:5
        fprintf('Segment #%d\n', ranked_segment);
        for dataset_idx = 1:length(dataset_names_all)   % loop over all datasets
            dataset_name = dataset_names_all{dataset_idx};
            dataset = dataset_dictionary(dataset_name);
            disp(dataset_name)
            % Specify model type (cpm or ridge), seed (defined above in
            % loop), feature_selection (ranked_20_percent,
            % ranked_10_percent, ranked_5_percent, ranked_1_percent),
            % ranked_segment (defined above in loop), null (1=null data for
            % permutation testing), control_covars (1=yes, covars defined
            % under datasets in this script)
            results = train_model_ranked_edges_percent(dataset_dictionary(dataset_name), 'model_type', 'cpm', 'seed', my_seed, 'num_folds', 10, 'feat_selection', 'ranked_10_percent', 'ranked_segment', ranked_segment, 'null', 0, "control_covars", 0);
            disp(results.r)
            results.dataset_name = dataset_name;
            r_2_decimals = sprintf('%.2f', results.r);
            save_name = [dataset_name, '_', results.model_type, '_pca_', results.pca_type, '_feat_', results.feat_selection, '_segment', num2str(results.ranked_segment), '_covary', num2str(results.control_covars), '_null', num2str(results.null), '_r', r_2_decimals, '_seed', num2str(results.seed), '.mat'];
            save(fullfile('/Users/brendan/Dropbox/analyses/ranked_edges/results/percent/cog_within_pca/cpm',save_name),'results');
        end
    end
end

