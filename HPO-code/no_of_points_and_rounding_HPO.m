%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Standalone Script to Optimize Preprocessing Parameters - Spinoff of
% Birgersson's original code for the J-V prediction project
%
% Purpose:
% This script systematically tests different preprocessing parameters to find
% the optimal combination for reconstructing IV curves. It iterates through
% various numbers of interpolation points and rounding precisions,
% calculates reconstruction accuracy (MAE, RMSE, R^2) for each, and
% generates summary plots to visualize the results.
%
% Date: Jun 2025
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;
tic; % Start a timer to see how long the whole process takes

%% --- 1. Configuration: Define Hyperparameters to Test ---

% Plausible combinations for [num_points_pre_mpp, num_points_post_mpp]
% Total points = pre + post - 1
mpp_combinations = { ...
    [2, 3], ... % 4 points
    [3, 3], ... % 5 points
    [3, 4], ... % 6 points (Original setting)
    [4, 4], ... % 7 points
    [4, 5], ... % 8 points
    [5, 5], ... % 9 points
    [5, 6], ... % 10 points
    [6, 6], ... % 11 points
    [6, 7], ... % 12 points
    [7, 8], ... % 14 points
    [8, 9]  ... % 16 points
};

% Number of significant digits to test for rounding
rounding_precisions = [2, 3, 4, 5, 6];

% Number of IV curves to sample for calculating statistics.
% Higher is more accurate but MUCH slower. 250 is a good balance.
N_samples_for_stats = 250; 

% Folder to save the final summary plots
output_folder = 'hyperparameter_optimization_results';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% --- 2. Data Loading (Done Once) ---
disp('Loading raw data...');
try
    current_density_raw = load('Data/Data_100k/iV_m.txt');
    input_params_raw = load('Data/Data_100k/LHS_parameters_m.txt');
    voltage_raw = [0:0.1:0.4, 0.425:0.025:1.4];
    disp('Data loaded successfully.');
catch ME
    error('Failed to load data. Make sure the "Data/Data_100k" folder is accessible. Error: %s', ME.message);
end

%% --- 3. Main Hyperparameter Sweep Loop ---

% Pre-allocate a table to store results
num_runs = length(mpp_combinations) * length(rounding_precisions);
result_vars = {'PreMPP', 'PostMPP', 'TotalPoints', 'SigDigits', 'AvgMAE', 'AvgRMSE', 'AvgR2'};
results = array2table(zeros(0, length(result_vars)), 'VariableNames', result_vars);
run_counter = 1;

fprintf('Starting hyperparameter sweep across %d combinations...\n', num_runs);
fprintf('This may take a while.\n\n');

for mpp_idx = 1:length(mpp_combinations)
    for round_idx = 1:length(rounding_precisions)
        
        % --- Get current parameters for this run ---
        current_mpp_combo = mpp_combinations{mpp_idx};
        num_points_pre_mpp = current_mpp_combo(1);
        num_points_post_mpp = current_mpp_combo(2);
        total_points = num_points_pre_mpp + num_points_post_mpp - 1;
        
        n_round = rounding_precisions(round_idx);
        
        fprintf('--> Run %d/%d: Pre=%d, Post=%d (Total=%d), Digits=%d\n', ...
            run_counter, num_runs, num_points_pre_mpp, num_points_post_mpp, total_points, n_round);
        
        % --- Core Processing Pipeline (re-run for each combination) ---
        
        % A. Interpolation and IV-Curve Reduction
        num_inputs = size(input_params_raw, 1);
        current_density_reduced = zeros(num_inputs, total_points);
        voltage_reduced = zeros(num_inputs, total_points);
        
        for idx = 1:num_inputs
            interp_voltage_range = linspace(0, 1.4, 1e4);
            current_density_interp = interp1(voltage_raw, current_density_raw(idx,:), interp_voltage_range, 'pchip');
            neg_index = find(current_density_interp < 0, 1);
            if ~isempty(neg_index)
                Voc = interp1(current_density_interp(neg_index-2:end), interp_voltage_range(neg_index-2:end), 0, 'linear');
            else
                v_extrap_range = linspace(0, 2, 1e4);
                i_extrap = interp1(voltage_raw, current_density_raw(idx,:), v_extrap_range, 'linear', 'extrap');
                neg_index = find(i_extrap < 0, 1);
                Voc = interp1(i_extrap(neg_index-2:end), v_extrap_range(neg_index-2:end), 0, 'linear');
            end
            power_interp = interp_voltage_range .* current_density_interp;
            [~, mpp_index] = max(power_interp);
            V_mpp = interp_voltage_range(mpp_index);
            voltage_pre_mpp = linspace(0, V_mpp, num_points_pre_mpp);
            voltage_post_mpp = linspace(V_mpp, Voc, num_points_post_mpp);
            current_pre_mpp = interp1(interp_voltage_range, current_density_interp, voltage_pre_mpp);
            current_post_mpp = interp1(interp_voltage_range, current_density_interp, voltage_post_mpp);
            voltage_reduced(idx,:) = [voltage_pre_mpp, voltage_post_mpp(2:end)];
            current_density_reduced(idx,:) = [current_pre_mpp, current_post_mpp(2:end)];
        end

        % B. Outlier Removal
        is_too_large = max(current_density_reduced, [], 2) > 1000;
        is_too_negative = min(current_density_reduced, [], 2) < -1;
        rows_to_remove_logical = is_too_large | is_too_negative;
        
        current_density_raw_clean = current_density_raw(~rows_to_remove_logical, :);
        voltage_reduced_clean = voltage_reduced(~rows_to_remove_logical, :);
        current_density_reduced_clean = current_density_reduced(~rows_to_remove_logical, :);

        % C. Set Voc to Zero & Rounding
        current_density_reduced_clean(:, end) = 0;
        voltage_reduced_clean = round_to_significant_digits(voltage_reduced_clean, n_round);
        current_density_reduced_clean = round_to_significant_digits(current_density_reduced_clean, n_round);
        
        % --- Performance Evaluation ---
        sample_indices = round(linspace(1, size(current_density_raw_clean, 1), N_samples_for_stats));
        all_mae = zeros(length(sample_indices), 1);
        all_rmse = zeros(length(sample_indices), 1);
        all_r_squared = zeros(length(sample_indices), 1);

        for i = 1:length(sample_indices)
            s_idx = sample_indices(i);
            
            known_voltages_raw = voltage_reduced_clean(s_idx, :);
            known_currents_raw = current_density_reduced_clean(s_idx, :);

            % --- FIX: Remove duplicate points caused by rounding ---
            [known_voltages, unique_indices] = unique(known_voltages_raw);
            known_currents = known_currents_raw(unique_indices);
            
            % If rounding reduces to a single point, we can't interpolate. Skip this sample.
            if length(known_voltages) < 2
                % Assign worst-case scores to penalize this combination
                all_mae(i) = Inf; 
                all_rmse(i) = Inf;
                all_r_squared(i) = -Inf;
                continue; % Skip to the next sample
            end
            % --- End of FIX ---

            Voc = max(known_voltages);
            
            relevant_indices = voltage_raw <= Voc & voltage_raw >= 0;
            voltage_for_stats = voltage_raw(relevant_indices);
            y_true = current_density_raw_clean(s_idx, relevant_indices);
            
            % Now, this interp1 call is safe
            y_pred = interp1(known_voltages, known_currents, voltage_for_stats, 'pchip');
            
            all_mae(i) = mean(abs(y_true - y_pred));
            all_rmse(i) = sqrt(mean((y_true - y_pred).^2));
            ss_res = sum((y_true - y_pred).^2);
            ss_tot = sum((y_true - mean(y_true)).^2);
            if ss_tot > 0, all_r_squared(i) = 1 - (ss_res / ss_tot); else, all_r_squared(i) = 1; end
        end
        
        % --- Store Results ---
        results(run_counter, :) = {num_points_pre_mpp, num_points_post_mpp, total_points, n_round, ...
                                   mean(all_mae), mean(all_rmse), mean(all_r_squared)};
        run_counter = run_counter + 1;
    end
end

%% --- 4. Results Analysis and Visualization ---
disp('Hyperparameter sweep complete. Generating summary plots...');

% Display the final results table
disp(' ');
disp('--- Final Performance Results Table ---');
disp(results);

% --- PLOT 1: Performance vs. Total Points (at fixed rounding) ---
fig1 = figure('Name', 'Performance vs. Number of Interpolation Points', 'Position', [100, 100, 900, 600]);
fixed_rounding = 4; % Analyze for 4 significant digits
subset = results(results.SigDigits == fixed_rounding, :);

colororder({'b', 'r'});
% Left Y-axis: RMSE
yyaxis left;
plot(subset.TotalPoints, subset.AvgRMSE, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
ylabel('Average RMSE (lower is better)');
xlabel('Total Number of Interpolation Points');
title(sprintf('Reconstruction Performance vs. Number of Points (at %d Sig. Digits)', fixed_rounding));
grid on;

% Right Y-axis: R-squared
yyaxis right;
plot(subset.TotalPoints, subset.AvgR2, '--s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
ylabel('Average R^2 (higher is better)');
ylim([min(subset.AvgR2) * 0.999, 1.0]); % Adjust R^2 axis to show small differences

legend('Avg. RMSE', 'Avg. R^2', 'Location', 'east');
saveas(fig1, fullfile(output_folder, 'performance_vs_points.png'));
disp('Saved plot: performance_vs_points.png');


% --- PLOT 2: Performance vs. Rounding Precision (at fixed points) ---
fig2 = figure('Name', 'Performance vs. Rounding Precision', 'Position', [150, 150, 900, 600]);
fixed_points_combo = mpp_combinations{3}; % Analyze for [3, 4] -> 6 total points
fixed_total_points = fixed_points_combo(1) + fixed_points_combo(2) - 1;
subset = results(results.TotalPoints == fixed_total_points, :);

colororder({'b', 'r'});
% Left Y-axis: RMSE
yyaxis left;
plot(subset.SigDigits, subset.AvgRMSE, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
ylabel('Average RMSE (lower is better)');
xlabel('Number of Significant Digits for Rounding');
title(sprintf('Reconstruction Performance vs. Rounding (at %d Total Points)', fixed_total_points));
grid on;

% Right Y-axis: R-squared
yyaxis right;
plot(subset.SigDigits, subset.AvgR2, '--s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
ylabel('Average R^2 (higher is better)');
ylim([min(subset.AvgR2) * 0.999, 1.0]);

legend('Avg. RMSE', 'Avg. R^2', 'Location', 'east');
saveas(fig2, fullfile(output_folder, 'performance_vs_rounding.png'));
disp('Saved plot: performance_vs_rounding.png');


toc; % Stop the timer and display elapsed time
disp('Analysis finished.');


%% --- Helper Function ---
function B = round_to_significant_digits(A, n)
    % Rounds each element of matrix A to n significant digits.
    B = zeros(size(A));
    for i = 1:numel(A)
        if A(i) ~= 0
            scaleFactor = 10^(n - ceil(log10(abs(A(i)))));
            B(i) = round(A(i) * scaleFactor) / scaleFactor;
        else
            B(i) = 0;
        end
    end
end