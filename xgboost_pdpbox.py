# 특정 feature의 끼치는 수치 경향을 알아보는 pdpbox
from pdpbox import pdp, info_plots, get_dataset

pima_data = dataset
pima_features = dataset.columns[:8]
pima_target = dataset.columns[8]

fig, axes, summary_df = info_plots.target_plot(
    df=pima_data, feature='Glucose', feature_name='Glucose', target=pima_target
)

# 특정 feature의 끼치는 수치 경향을 알아보는 캔들스틱 차트
fig, axes, summary_df = info_plots.actual_plot(
    model=model, 
    X=pima_data[pima_features], 
    feature='Glucose', 
    feature_name='Glucose', 
    predict_kwds={}
)

pdp_gc = pdp.pdp_isolate(
    model=model, 
    dataset=pima_data, 
    model_features=pima_features,
    feature='Glucose'
)

#  plot information
fig, axes = pdp.pdp_plot(
    pdp_gc, 
    'Glucose', 
    plot_lines=False, 
    frac_to_plot=0.5, 
    plot_pts_dist=True
)

pdp_gc = pdp.pdp_isolate(
    model=model, 
    dataset=pima_data, 
    model_features=pima_features,
    feature='Glucose'
)

#  plot information
fig, axes = pdp.pdp_plot(
    pdp_gc, 
    'Glucose', 
    plot_lines=True, 
    frac_to_plot=0.5, 
    plot_pts_dist=True
)

# calculate model with BloodPressure to express pdp
pdp_bp = pdp.pdp_isolate(
    model=model, 
    dataset=pima_data, 
    model_features=pima_features, 
    feature='BloodPressure'
)
# plot PDP on BloodPressure
fig, axes = pdp.pdp_plot(pdp_bp, 
                         'BloodPressure', 
                         plot_lines=False, 
                         frac_to_plot=0.5, 
                         plot_pts_dist=True)


pdp_intertatcion = pdp.pdp_interact(
    model=model, 
    dataset=pima_data, 
    model_features=pima_features, 
    features=['BloodPressure', 'Glucose']
)

fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_intertatcion, 
    feature_names=['BloodPressure', 'Glucose'], 
    plot_type='contour', 
    x_quantile=True, 
    plot_pdp=True
)

fig, axes = pdp.pdp_interact_plot(pdp_intertatcion, 
                                  ['BloodPressure', 'Glucose'], 
                                  plot_type='grid', 
                                  x_quantile=True, 
                                  plot_pdp=False)