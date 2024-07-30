import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import re
import shap
from matplotlib.ticker import MaxNLocator, FuncFormatter


def get_patch_files(folder_path):
    files = os.listdir(folder_path)
    patch_files = {}
    for file in files:
        patch_number = file.split('_')[2]  # Assuming filename format 'soy_patch_X_...'
        if patch_number not in patch_files:
            patch_files[patch_number] = []
        patch_files[patch_number].append(file)
    return patch_files

def plot_vwc_comparison(actual, predicted_RF, predicted_FM,predicted_FM2,doy, patch_number, title='VWC Comparison'):
    plt.figure(figsize=(14, 8))
    plt.plot(doy, actual, label='Actual Height', color='#0000CD', linestyle='-', marker='o', markersize=10, linewidth=3)
    
    # Plot the first model's predictions with a dashed line and triangle markers
    plt.plot(doy, predicted_RF, label='Estimated Height-RF', color='#FFA500', linestyle='--', marker='^', markersize=10, linewidth=3)
    
    # Plot the second model's predictions with a dash-dot line and square markers
    plt.plot(doy, predicted_FM, label='Estimated Height STL-FM', color='#228B22', linestyle='-.', marker='s', markersize=10, linewidth=3)
    
    # Plot the third model's predictions with a dotted line and diamond markers
    plt.plot(doy, predicted_FM2, label='Estimated Height MTL-FM', color='#FF1493', linestyle=':', marker='D', markersize=10, linewidth=3)
    plt.title(f'{title} - Patch {patch_number}', fontsize=16)  # Adjusted fontsize for readability
    plt.xlabel('Day of Year', fontsize=16)
    plt.ylabel('Height (cm)', fontsize=16)  # Corrected m2 to m^2 for proper notation
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=4, fontsize=30)
    plt.legend(loc='best', shadow=True, fontsize=30)
    plt.xticks(fontsize=16)  # Adjusted fontsize for better visibility
    plt.yticks(fontsize=16)  # Adjusted fontsize for better visibility

    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.tick_params(axis='both', which='major', labelsize=30, width=3, length=16)  # Adjusted tick parameters for visibility
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)


    plt.tight_layout()  # Ensure layout fits without overlap
    plt.show()

def plot_vwc_scatter(ax, actual, predicted_RF, predicted_FM, predicted_FM2, doy, patch_number, title='VWC Comparison'):
    # Base position for the day of the year
    base_doy = 225
    
    # Create offsets for each type of measurement
    doy_offsets = {
        'Actual Height': 225,
        'Estimated Height-RF': 225+0.5,
        'Estimated Height ST-FM': 225+0.75,
        'Estimated Height MT-FM': 225+1
    }

    # Plotting each type of measurement with its own offset
    ax.scatter(doy_offsets['Actual Height'], actual, label='Actual Height', color='#0000CD', s=200, edgecolor='k')
    ax.scatter(doy_offsets['Estimated Height-RF'], predicted_RF, label='Estimated Height-RF', color='#FFA500', s=200, edgecolor='k')
    ax.scatter(doy_offsets['Estimated Height ST-FM'], predicted_FM, label='Estimated Height ST-FM', color='#228B22', s=200, edgecolor='k')
    ax.scatter(doy_offsets['Estimated Height MT-FM'], predicted_FM2, label='Estimated Height MT-FM', color='#FF1493', s=200, edgecolor='k')

    # Setting plot titles and labels
    ax.set_title(title)
    ax.set_xlabel('Day of Year', fontsize=30)
    ax.set_ylabel('Height (cm)', fontsize=30)
    ax.grid(True)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=30, width=3, length=16)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    # Set the x-axis range based on the minimum and maximum offsets
    ax.set_xlim([225 - 1, 226 + 1])


def load_data_for_patches(folder_path, patches):
    data_list = []
    label_list = []
    doy_list = []
    patch_indices = []

    for file_name in patches:
        if file_name.endswith('_data.npy'):
            data_path = os.path.join(folder_path, file_name)
            patch_number = int(re.search(r'_patch_(\d+)_', data_path).group(1))

            label_path = os.path.join(folder_path, file_name.replace('_data.npy', '_label.npy'))
            SLC_path = os.path.join(folder_path, file_name.replace('_data.npy', '_SLC.npy'))
            S2_path = os.path.join(folder_path, file_name.replace('_data.npy', '_S2_2.npy'))
            climate_path = os.path.join(folder_path, file_name.replace('_data.npy', '_climate.npy'))

            if os.path.exists(data_path) and os.path.exists(label_path) and os.path.exists(SLC_path) and os.path.exists(S2_path):
                data = np.load(data_path)
                label = np.load(label_path)
                SLC = np.load(SLC_path)
                new_S2 = np.load(S2_path)
                climate = np.load(climate_path)

                if patch_number <= 6:
                    data2 = data[0, 1:6, 5:15]
                    label2 = label[0, 2, 5:15]
                    SLC2 = SLC[0, 1:, 2:]
                    new_S2 = new_S2[1:, 1:]
                    climate = climate[0, 1:, 1:]
                    doy = data[0, 0, 5:15]
                elif patch_number <= 14 and patch_number > 6:
                    data2 = data[0, 1:6, 3:13]
                    label2 = label[0, 2, 3:13]
                    SLC2 = SLC[0, 1:, 3:]
                    new_S2 = new_S2[0, 1:, :-1]
                    climate = climate[0, 1:, :]
                    doy = data[0, 0, 3:13]
                elif patch_number == 16:
                    data2 = data[0, 1:6, 3:13]
                    label2 = label[0, 2, 3:13]
                    SLC2 = SLC[0, 1:, 3:]
                    new_S2 = new_S2[1:,:]
                    climate = climate[0, 1:, :]
                    doy = data[0, 0, 3:13]
                else:
                    data2 = data[0, 1:6]
                    data2 = data2.reshape(-1, 1)
                    label2 = label[0, 1, :]
                    SLC2 = SLC[0, 1:, :]
                    SLC2 = SLC2.reshape(-1, 1)
                    new_S2 = new_S2[1:].reshape(-1, 1)
                    climate = climate[:, np.newaxis]
                    doy = data[0, 0]

                combined_data = np.concatenate([data2[0:2],new_S2,climate], axis=0)
                num_data_points = len(label2)
                patch_indices.extend([patch_number] * num_data_points)

                data_list.append(combined_data)
                label_list.append(label2)
                doy_list.append(doy.reshape(-1, 1))
      

    data_array = np.concatenate(data_list, axis=1).transpose()
    labels_array = np.concatenate(label_list, axis=0)
    print(doy_list)
    doy_array = np.concatenate(doy_list, axis=0)

    return data_array, labels_array, doy_array, np.array(patch_indices)

def calculate_metrics(true_values, predicted_values):
    mae = np.mean(np.abs(predicted_values - true_values))
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    return mae, rmse, r2

def plot_feature_importance(importances, feature_names, title):
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 8), dpi=100)
    plt.title(title, fontsize=18)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=16)
    plt.xlabel('Relative Importance', fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# Main execution
current_directory = os.path.dirname(os.path.realpath(__file__))
train_folder_path = os.path.join(current_directory, 'soy_nonirrigated')
test_folder_path = os.path.join(current_directory, 'soy_nonirrigated_test')

train_patch_files = get_patch_files(train_folder_path)
test_patch_files = get_patch_files(test_folder_path)

train_patches = list(train_patch_files.keys())
test_patches = list(test_patch_files.keys())

train_files = [file for patch in train_patches for file in train_patch_files[patch]]
test_files = [file for patch in test_patches for file in test_patch_files[patch]]

train_data, train_labels, train_doy, train_patch_indices = load_data_for_patches(train_folder_path, train_files)
test_data, test_labels, test_doy, test_patch_indices = load_data_for_patches(test_folder_path, test_files)

imputer = SimpleImputer(strategy='mean')
train_data = imputer.fit_transform(train_data)
test_data = imputer.transform(test_data)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# K-fold cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rf_metrics = []
xgb_metrics = []

for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
    print(f"Fold {fold + 1}/{k}")
    
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=1, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=3, colsample_bytree=0.7, subsample=0.7, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    
    rf_fold_metrics = calculate_metrics(y_val, rf_pred)
    rf_metrics.append(rf_fold_metrics)
    
    xgb_fold_metrics = calculate_metrics(y_val, xgb_pred)
    xgb_metrics.append(xgb_fold_metrics)
    
    print(f"RF Metrics: {rf_fold_metrics}")
    print(f"XGB Metrics: {xgb_fold_metrics}")

rf_avg_metrics = np.mean(rf_metrics, axis=0)
xgb_avg_metrics = np.mean(xgb_metrics, axis=0)

print("\nAverage Random Forest Metrics:", rf_avg_metrics)
print("Average XGBoost Metrics:", xgb_avg_metrics)

# Train final models
rf_final = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=1, min_samples_split=5, random_state=42)
rf_final.fit(train_data, train_labels)

xgb_final = xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=3, colsample_bytree=0.7, subsample=0.7, random_state=42)
xgb_final.fit(train_data, train_labels)

# Test set predictions
rf_test_pred = rf_final.predict(test_data)
xgb_test_pred = xgb_final.predict(test_data)



#regulaiZATION

height= [ 14.1058445,  22.395214 ,  37.231365 ,  43.963215 ,  76.347565  , 96.269455,
 104.21892 ,  104.37649  ,  83.51968 ,   45.33398 ,   15.848012,   20.292492,
  32.520676,   41.58614  ,  66.468636 ,  84.06462 ,   92.870964 ,  93.5928,
  79.107834  , 52.297318 ,  21.123953  , 33.51735  ,  65.08072  ,  87.31014,
 102.956474 , 102.93503 ,  100.68888   , 80.122894 ,  73.04811  ,  -5.1237106,
  25.942177  , 41.157616 ,  56.816708  , 75.226105 ,  97.944244 ,  95.967896,
  95.26361   , 87.898994 ,  77.4246    ,  1.2808186,  19.350967 ,  37.13673,
  55.410297  , 79.90916   , 87.336044  , 94.654655,  101.43568 ,  110.58053,
  59.56012   , -2.3431544 ,  9.249552  ,  7.944204,  126.16034 ,  115.20021,
 118.86044   ,128.25014  , 117.96451  , 131.3795 ,   133.00793 ,  125.98634  ]


height= [ 13.504 ,      21.719553 ,   34.70921  ,   40.4533  ,    73.01633,
  92.40466,     99.65046  ,   99.061615 ,   80.03419  ,   48.00296,
  19.87432 ,    23.110943 ,   34.9116 ,     45.73778  ,   72.10941,
  90.434235 ,   99.29698  ,  100.3572  ,    84.59039 ,    55.842983,
  19.044703 ,   30.470207 ,   59.63691  ,   82.994736 ,   97.996475,
 100.13296   ,  98.291046 ,   77.68731  ,   66.85901  ,    0.26765954,
  22.884943  ,  35.640564 ,   49.3329   ,   67.124664 ,   87.69522,
  89.460045  ,  88.808365,    80.33004  ,   73.74032  ,    4.3492002,
  38.257183  ,  61.85942,     90.18542  ,  103.38251 ,   103.463715,
 103.92864   , 110.39197,    123.72135  ,   65.98065 ,     0.6189618,
   7.0807185  ,  6.0953546,  117.02388  ,  108.42455,    112.459114,
 121.90564   , 110.0015  ,   124.72785  ,  123.05051 ,   116.66109   ] 

FM_test_pred=[ 25.223265 , 23.802841 , 29.498568 , 51.3621  ,  69.643585,  91.90562,
 109.97265  ,101.25406  , 89.75928  , 51.10519 ,  30.314606 , 26.381601,
  31.391829 , 54.28186 ,  71.25387  , 92.54456 , 108.47063  ,102.61312,
  88.066475 , 65.1915 ,   28.227095 , 41.454018 , 61.590015 , 80.15224,
 103.36545 ,  96.326546, 100.276306 , 72.248184, 91.70235  , 13.635659,
  24.328829 , 32.08767 ,  47.16563  , 54.29883,   84.82572 ,  86.98487,
  92.17442 ,  74.53049 ,  82.89156  , 16.14622,   48.182583 , 29.31907,
  56.694344 , 74.373116 , 90.39532  , 87.552895, 103.748474, 102.89044,
  81.10263  , 16.198551 , 10.984295 ,  7.057986 ,116.86459 , 118.86937,
 115.748886 ,120.83014 , 116.80162 , 118.10226 , 122.07731 , 116.85679 ]


FM_test_pred=np.array(FM_test_pred)

height=np.array(height)

print('rf',rf_test_pred)

rf_test_metrics = calculate_metrics(test_labels, rf_test_pred)
xgb_test_metrics = calculate_metrics(test_labels, xgb_test_pred)

print("\nFinal Random Forest Test Metrics:", rf_test_metrics)
print("Final XGBoost Test Metrics:", xgb_test_metrics)

# Feature importance and plotting
feature_names = ['VH', 'VV', 'VH/VV', 'RVI', 'Incidence angle', 'Entropy', 'Anisotropy', 'Alpha' ,'Red-edge', 'NDVI', 'NDWI', 'P', 'Tmin', 'Tmax']
rf_feature_importance = rf_final.feature_importances_
xgb_feature_importance = xgb_final.feature_importances_

plot_feature_importance(rf_feature_importance, feature_names, 'Feature Importances - Random Forest')
plot_feature_importance(xgb_feature_importance, feature_names, 'Feature Importances - XGBoost')

# SHAP analysis
rf_explainer = shap.TreeExplainer(rf_final)
xgb_explainer = shap.TreeExplainer(xgb_final)

rf_shap_values = rf_explainer.shap_values(train_data)
xgb_shap_values = xgb_explainer.shap_values(train_data)

shap.summary_plot(rf_shap_values, train_data, feature_names=feature_names, plot_type="bar")
shap.summary_plot(xgb_shap_values, train_data, feature_names=feature_names, plot_type="bar")

plt.style.use('default')
plt.rcParams.update({'font.size': 12})  # Adjust this value to change the overall font size

# Create a figure with a specific size
fig, ax = plt.subplots(figsize=(30, 20), dpi=300)

# Generate the SHAP summary plot
shap.summary_plot(
    rf_shap_values,  # or xgb_shap_values
    train_data,
    feature_names=feature_names,
    plot_type="bar",
    color='#86bf91',  # Change this to your desired color
    axis_color='black',
    show=False  # Don't display the plot yet
)

# Customize the plot
ax = plt.gca()  # Get the current axis
ax.tick_params(axis='both', which='major', labelsize=20)  # Adjust tick label font size
ax.set_xlabel('mean(|SHAP value|)', fontsize=20)  # Adjust x-label font size
#ax.set_ylabel('Feature', fontsize=20)  # Adjust y-label font size

plt.savefig('VWC_soy.png')

plt.tight_layout()
plt.show()


# Plot VWC comparisons for each patch
for patch in np.unique(test_patch_indices):
    patch_mask = test_patch_indices == patch
    patch_pred_rf = rf_test_pred[patch_mask]
    patch_pred_FM=FM_test_pred[patch_mask]
    patch_pred_FM2=height[patch_mask]
    patch_pred_xgb = xgb_test_pred[patch_mask]
    patch_label = test_labels[patch_mask]
    patch_doy = test_doy[patch_mask]
    if patch<25:

        plot_vwc_comparison(patch_label, patch_pred_rf,patch_pred_FM,patch_pred_FM2, patch_doy, patch, title='Random Forest')
        #plot_vwc_comparison(patch_label, patch_pred_xgb, patch_doy, patch, title='XGBoost')
    else:
        print('hellp')
     
# =============================================================================
# fig, ax = plt.subplots(figsize=(8, 16),dpi=300)
# title = 'Random Forest - Combined Patches'
# 
# for patch in np.unique(test_patch_indices):
#     
#     patch_mask = test_patch_indices == patch
#     patch_actual = test_labels[patch_mask]
#     patch_pred_rf = rf_test_pred[patch_mask]
#     patch_pred_fm = FM_test_pred[patch_mask]
#     patch_pred_FM2=height[patch_mask]
#     patch_doy = test_doy[patch_mask]
#     if patch>25:
#         if patch_doy<200:
#             
#     # Call plot function without any restrictive condition, or handle it inside the function
#             plot_vwc_scatter(ax, patch_actual, patch_pred_rf, patch_pred_fm,patch_pred_FM2, patch_doy, patch, title)
# 
# # Handle legend and show plot outside the loop
# #x.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
# 
# =============================================================================
        