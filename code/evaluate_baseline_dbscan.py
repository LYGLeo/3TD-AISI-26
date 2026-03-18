import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import traceback

# Suppress UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# =======================================
# 1. Configuration
# =======================================
# --- Experiment Settings ---
MODELS_TO_RUN = ['LightGBM']
STRIDES_TO_RUN = [5]
USERS_TO_RUN = None

# --- Path Settings ---
# Top-level directory for probability data
PROB_DATA_DIR = "/home/yglee/evdp/results/baseline/passive/"
# Path to the CSV file with per-user quantile boundary data
BOUNDARY_CSV_PATH = "/home/yglee/evdp/results/baseline/historical/final_predictions_ALL_MODELS.csv"
# Path to the CSV file containing weekday/weekend information
DOW_CSV_PATH = "/home/yglee/Dataset/STDD_Data/stride#5/dow_labeled.csv"
# Directory to save final results (tables, plots)
RESULTS_DIR = "/home/yglee/evdp/results/baseline/passive/"

# --- DBSCAN and Boundary Parameters ---
DBSCAN_PARAMS = {
    'min_samples_start': 3,    # Minimum data points to attempt DBSCAN
    'prob_threshold': 0.2,     # Minimum probability for a point to be considered a detection
    'spike_threshold': 0.05    # Probability must increase by this much from the previous value
}
# Threshold to force a departure detection regardless of other conditions
FORCE_DEPARTURE_PROB = 0.5

# Quantile column names for weekday boundaries
WEEKDAY_LOWER_COL = 'pred_hour_LGBM_Q45'
WEEKDAY_UPPER_COL = 'pred_hour_LGBM_Q55'

# Quantile column names for weekend boundaries
WEEKEND_LOWER_COL = 'pred_hour_LGBM_Q45'
WEEKEND_UPPER_COL = 'pred_hour_LGBM_Q55'


# =======================================
# 2. Helper Functions
# =======================================
def _estimate_eps(prob_list, min_pts, eps_threshold=0.01):
    """Estimates the optimal eps value for DBSCAN based on a probability list."""
    if len(prob_list) < min_pts:
        return eps_threshold
    
    data = np.array(prob_list).reshape(-1, 1)
    
    if (data.max() - data.min()) < 1e-9:
        return eps_threshold
        
    neigh = NearestNeighbors(n_neighbors=min_pts).fit(data)
    dists, _ = neigh.kneighbors(data)
    k_dist = np.sort(dists[:, -1])
    
    try:
        kneedle = KneeLocator(x=range(len(k_dist)), y=k_dist, S=1.0, curve="convex", direction="increasing")
        eps = kneedle.elbow_y if kneedle.elbow_y is not None else k_dist[-1]
    except (ValueError, TypeError):
        eps = k_dist[-1]
        
    return max(eps, eps_threshold)


# =======================================
# 3. Main Analysis Function
# =======================================
def analyze_departure_with_dbscan(day_prob_df, day_boundary_row, params):
    """
    Analyzes departure time for a single user's daily data using DBSCAN.
    (Improved logic: pre-filters data based on a 'true prediction time'.)
    """
    # --- 3.1. Define 'true prediction time' and set boundaries ---
    day_prob_df['pred_time'] = day_prob_df['window_end'] + pd.Timedelta(minutes=30)
    
    day_ts = pd.Timestamp(day_prob_df['date'].iloc[0]).tz_localize(day_prob_df['pred_time'].dt.tz)
    
    lower_h = day_boundary_row[params['lower_col']]
    upper_h = day_boundary_row[params['upper_col']]
    
    lower_bound_ts = day_ts.replace(hour=int(lower_h), minute=int((lower_h * 60) % 60), second=0, microsecond=0)
    upper_bound_ts = day_ts.replace(hour=int(upper_h), minute=int((upper_h * 60) % 60), second=0, microsecond=0)

    # --- 3.2. Filter data for analysis ---
    analysis_df = day_prob_df[
        (day_prob_df['pred_time'] >= lower_bound_ts) & 
        (day_prob_df['pred_time'] <= upper_bound_ts)
    ].copy()

    if analysis_df.empty:
        pred_time = upper_bound_ts
        detection_method = 'upper_bound_forced (no candidates)'
    else:
        prob_history = []
        
        # Find the index of the first row in analysis_df within the original day_prob_df
        first_analysis_idx = analysis_df.index[0]
        loc_in_original = day_prob_df.index.get_loc(first_analysis_idx)
        
        # Get padding data from before the analysis window
        num_padding = params['min_samples_start'] - 1
        padding_start_loc = max(0, loc_in_original - num_padding)
        
        if padding_start_loc < loc_in_original:
            padding_df = day_prob_df.iloc[padding_start_loc:loc_in_original]
            prob_history = padding_df['prob'].tolist()

        # --- 3.3. Departure Detection Logic ---
        pred_time = pd.NaT
        detection_method = 'no_detection'
        
        for idx, row in analysis_df.iterrows():
            prob_history.append(row['prob'])
            
            current_prediction_time = row['pred_time']
            
            # Condition 1: Force Departure (based on probability)
            if row['prob'] >= params['force_departure_prob']:
                pred_time = current_prediction_time
                detection_method = 'force_prob'
                break

            # Condition 2: DBSCAN
            current_min_pts = params['min_samples_start']
            while current_min_pts <= len(prob_history):
                eps = _estimate_eps(prob_history, current_min_pts)
                if eps > 0:
                    db = DBSCAN(eps=eps, min_samples=current_min_pts).fit(np.array(prob_history).reshape(-1, 1))
                    is_outlier = (db.labels_[-1] == -1)
                    is_above_threshold = (row['prob'] >= params['prob_threshold'])
                    is_spike = len(prob_history) > 1 and (row['prob'] > prob_history[-2] + params['spike_threshold'])
                    
                    if is_outlier and is_above_threshold and is_spike:
                        pred_time = current_prediction_time
                        detection_method = 'dbscan'
                        break
                current_min_pts += 1
            
            if pd.notna(pred_time):
                break

        # --- 3.4. Fallback Logic ---
        if pd.isna(pred_time):
            pred_time = upper_bound_ts
            detection_method = 'upper_bound_forced'

    # --- 3.5. Return Results ---
    mae = abs((pred_time - day_prob_df['departure_time'].iloc[0]).total_seconds()) / 60
    
    return {
        'date': day_prob_df['date'].iloc[0],
        'predicted_departure': pred_time,
        'true_departure': day_prob_df['departure_time'].iloc[0],
        'mae_minutes': mae,
        'is_weekend': day_prob_df['is_weekend'].iloc[0],
        'detection_method': detection_method,
        'lower_bound_hour': lower_h,
        'upper_bound_hour': upper_h,
        'analysis_start_time': analysis_df['window_end'].iloc[0] if not analysis_df.empty else pd.NaT,
        'analysis_end_time': analysis_df['window_end'].iloc[-1] if not analysis_df.empty else pd.NaT
    }


# =======================================
# 4. Visualization Function
# =======================================
def plot_daily_predictions(day_prob_df, analysis_result, save_path):
    """Visualizes daily analysis results and saves the plot."""
    day = analysis_result['date']
    fig, ax = plt.subplots(figsize=(15, 6))

    # Remove timezone information for plotting
    plot_times = (day_prob_df['window_end'] + pd.Timedelta(minutes=30)).dt.tz_localize(None)

    ax.plot(plot_times, day_prob_df['prob'], marker='o', ms=4, linestyle='-', label='Departure Probability', zorder=3)

    # Create boundary area with timezone-naive timestamps
    day_naive = pd.Timestamp(day)
    lower_bound_dt = day_naive.replace(hour=int(analysis_result['lower_bound_hour']), minute=int((analysis_result['lower_bound_hour'] * 60) % 60))
    upper_bound_dt = day_naive.replace(hour=int(analysis_result['upper_bound_hour']), minute=int((analysis_result['upper_bound_hour'] * 60) % 60))
    ax.axvspan(lower_bound_dt, upper_bound_dt, color='grey', alpha=0.2, label=f"Boundary ({analysis_result['lower_bound_hour']:.2f}h - {analysis_result['upper_bound_hour']:.2f}h)")

    # Remove timezone info from true/predicted departure times for plotting
    true_departure = analysis_result['true_departure'].tz_localize(None)
    ax.axvline(x=true_departure, color='black', linestyle='-', lw=2.5, label=f"True: {true_departure.strftime('%H:%M')}", zorder=4)
    if pd.notna(analysis_result['predicted_departure']):
        predicted_departure = analysis_result['predicted_departure'].tz_localize(None)
        ax.axvline(x=predicted_departure, color='blue', linestyle='-.', lw=2.5, label=f"Predicted: {predicted_departure.strftime('%H:%M')}", zorder=4)

    title = (f"User {day_prob_df['user_id'].iloc[0]} - {day.strftime('%Y-%m-%d')} "
              f"(MAE: {analysis_result['mae_minutes']:.1f} min, Method: {analysis_result['detection_method']})")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    # Set x-axis range with timezone-naive timestamps
    start_time = day_naive.replace(hour=0, minute=0, second=0)
    end_time = start_time + pd.Timedelta(days=1)
    ax.set_xlim(start_time, end_time)

    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle=':')
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =======================================
# 5. Full Pipeline Execution
# =======================================
def run_analysis_pipeline():
    """Runs the full analysis pipeline sequentially."""
    print("="*50)
    print("🚀 Starting DBSCAN Analysis Pipeline")
    print("="*50)

    boundary_df = pd.read_csv(BOUNDARY_CSV_PATH, parse_dates=['date'])
    dow_df = pd.read_csv(DOW_CSV_PATH, parse_dates=['date'])
    dow_df['date'] = dow_df['date'].dt.normalize()

    all_results = []

    for model_name in tqdm(MODELS_TO_RUN, desc="Total Models"):
        for stride in tqdm(STRIDES_TO_RUN, desc=f"Strides for {model_name}", leave=False):
            prob_dir = os.path.join(PROB_DATA_DIR, model_name, f"stride_{stride}")
            prob_files = glob(os.path.join(prob_dir, "probs_generalized_*.csv"))
            
            for prob_file_path in tqdm(prob_files, desc="Processing Files", leave=False):
                try:
                    prob_df_all_users = pd.read_csv(prob_file_path, parse_dates=['window_end', 'departure_time', 'date'])
                    
                    original_rows = len(prob_df_all_users)
                    prob_df_all_users['date'] = prob_df_all_users['date'].dt.normalize()
                    prob_df_all_users = pd.merge(prob_df_all_users, dow_df, on='date', how='left')

                    users_in_file = prob_df_all_users['user_id'].unique()
                    if USERS_TO_RUN is not None:
                        users_to_process = [u for u in users_in_file if u in USERS_TO_RUN]
                    else:
                        users_to_process = users_in_file

                    for user_id in tqdm(users_to_process, desc=f"Users in {os.path.basename(prob_file_path)}", leave=False):
                        prob_df = prob_df_all_users[prob_df_all_users['user_id'] == user_id].copy()
                        if prob_df.empty:
                            continue
                        
                        user_boundary_df = boundary_df[boundary_df['user_id'] == user_id]
                        if user_boundary_df.empty:
                            continue
                        
                        user_day_results = []
                        for day_val, day_prob_df in prob_df.groupby(prob_df['date'].dt.date):
                            day_boundary_row = user_boundary_df[user_boundary_df['date'].dt.date == day_val]
                            if day_boundary_row.empty:
                                continue
                            
                            is_weekend = day_prob_df['is_weekend'].iloc[0] == 1
                            lower_col = WEEKEND_LOWER_COL if is_weekend else WEEKDAY_LOWER_COL
                            upper_col = WEEKEND_UPPER_COL if is_weekend else WEEKDAY_UPPER_COL

                            analysis_params = {
                                'lower_col': lower_col,
                                'upper_col': upper_col,
                                'force_departure_prob': FORCE_DEPARTURE_PROB,
                                **DBSCAN_PARAMS
                            }
                            
                            result = analyze_departure_with_dbscan(day_prob_df.reset_index(drop=True), day_boundary_row.iloc[0], analysis_params)
                            
                            if result:
                                user_day_results.append(result)
                                lower_q = lower_col.split('Q')[-1]
                                upper_q = upper_col.split('Q')[-1]
                                prefix = "WE" if is_weekend else "WD"
                                quantile_folder = f"{prefix}_Q{lower_q}_Q{upper_q}"

                                plot_save_dir = os.path.join(
                                    RESULTS_DIR, "plots", model_name, f"stride_{stride}", quantile_folder, f"user_{user_id}"
                                )
                                os.makedirs(plot_save_dir, exist_ok=True)
                                plot_save_path = os.path.join(plot_save_dir, f"plot_{day_val.strftime('%Y-%m-%d')}.png")
                                
                                plot_daily_predictions(day_prob_df, result, plot_save_path)
                        
                        if user_day_results:
                            user_results_df = pd.DataFrame(user_day_results)
                            user_results_df['user_id'] = user_id
                            user_results_df['model'] = model_name
                            user_results_df['stride'] = stride
                            all_results.append(user_results_df)

                except Exception as e:
                    print(f"\n❌ ERROR processing {prob_file_path} for user {user_id}: {e}")
                    traceback.print_exc()

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)

        wd_param_str = f"WD_Q{WEEKDAY_LOWER_COL.split('Q')[-1]}_Q{WEEKDAY_UPPER_COL.split('Q')[-1]}"
        we_param_str = f"WE_Q{WEEKEND_LOWER_COL.split('Q')[-1]}_Q{WEEKEND_UPPER_COL.split('Q')[-1]}"
        stride_str = f"stride{STRIDES_TO_RUN[0]}" if len(STRIDES_TO_RUN) == 1 else "multi_stride"
        model_str = f"{MODELS_TO_RUN[0]}" if len(MODELS_TO_RUN) == 1 else "multi_model"

        filename = f"final_summary_{model_str}_{wd_param_str}_{we_param_str}_{stride_str}_0.5.csv"
        final_save_path = os.path.join(RESULTS_DIR, "tables", filename)
        
        final_df.to_csv(final_save_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ Analysis complete. Final summary saved to: {final_save_path}")
    else:
        print("\n⏹️ Analysis finished, but no results were generated.")


if __name__ == '__main__':
    run_analysis_pipeline()