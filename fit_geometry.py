from dd_turb_design import fit_data, read_in_large_dataset, split_data
from joblib import dump,load



data = read_in_large_dataset(state_retention_statistics=True)

traindf,testdf=split_data(data)

fit = fit_data(traindf,
                variables=['phi',
                            'psi',
                            'M2',
                            'Co'],
                output_key='stagger_stator',
                number_of_restarts=0,           
                length_bounds=[1e-3,1e2],
                noise_magnitude=1e-3,
                noise_bounds=[1e-6,1e-1],
                nu='optimise')
print(fit.optimised_kernel)

load('fitted_test_1.joblib')

# fit.plot(x1='phi')

# fit.plot_accuracy(testdf,
#                   line_error_percent=10)