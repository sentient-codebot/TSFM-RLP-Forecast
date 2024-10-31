import matplotlib.pyplot as plt
import os

def plot_predictions_point(x, y, y_hat, country, reso, _type, model_name, num_steps_day, _path):
    x = x.reshape(-1, num_steps_day * 3)
    target_range = range(num_steps_day * 3, num_steps_day * 4)

    plt.figure(figsize=(6, 4))
    plt.plot(range(num_steps_day * 3), x[-1, :], label='Input', color='b', linewidth=1.5)
    plt.plot(target_range, y[-1, :], label='Target', color='r', linewidth=1.5)
    plt.plot(target_range, y_hat[-1, :], label='Prediction (Mean)', color='g', linewidth=1.5)

    plt.xlabel('Time [Hour]', fontsize=14)
    plt.ylabel('Electricity Consunption [kWh]', fontsize=14)
    plt.title(f'{model_name} predictions for {country.capitalize()}-{_type.capitalize()}-{reso}', fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(fontsize=12, loc='upper left')
    
    # Saving the plot
    plt.savefig(f'{_path}/{model_name}_{country}_{reso}_{_type}.png', bbox_inches='tight')
    plt.close()


def plot_gp_predictions(x, y, mean, low, high, country, reso, _type,  _path = 'exp/gp/result/'):
    plt.figure(figsize=(6, 4))
    
    # Plot the input sequence
    plt.plot(x[-1, :], label='Input', color='b')
    
    # Create the range for the target and predicted values
    target_range = range(len(x[-1, :]), len(x[-1, :]) + len(y[0, :]))
    
    # Plot the target sequence
    plt.plot(target_range, y[-1, :], color='r', label='Target')
    
    # Plot the median prediction
    plt.plot(target_range, mean[-1, :], color='g', label='Prediction (Mean)')
    
    # Fill the area between low and high predictions to show uncertainty
    plt.fill_between(target_range, low[-1, :], high[-1, :], color='gray', alpha=0.3, label='Uncertainty')
    
    # Set plot labels and title
    plt.xlabel('Time [Hour]', fontsize=14)
    plt.ylabel('Electricity Consunption [kWh]', fontsize=14)
    plt.title(f'GP Predictions for {country.capitalize()}-{_type.capitalize()}-{reso}', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')
    
    # Save the plot
    os.makedirs(_path, exist_ok=True)
    plt.savefig(os.path.join(_path, f'gp_{country}_{reso}_{_type}.png'), bbox_inches='tight')
    plt.close()



def plot_chronos_predictions(_input, _target, median, low, high, country, reso, _type, _path = 'exp/chronos_exp/result/'):
    plt.figure(figsize=(6, 4))
    
    # Plot the input sequence
    plt.plot(_input[0, :], label='Input', color='b')
    
    # Create the range for the target and predicted values
    target_range = range(len(_input[0, :]), len(_input[0, :]) + len(_target[0, :]))
    
    # Plot the target sequence
    plt.plot(target_range, _target[0, :], color='r', label='Target')
    
    # Plot the median prediction
    plt.plot(target_range, median[0, :], color='g', label='Prediction (Mean)')
    
    # Fill the area between low and high predictions to show uncertainty
    plt.fill_between(target_range, low[0, :], high[0, :], color='gray', alpha=0.3, label='Uncertainty')
    
    # Set plot labels and title
    plt.xlabel('Time [Hour]', fontsize=14)
    plt.ylabel('Electricity Consunption [kWh]', fontsize=14)
    plt.title(f'Chronos Predictions for {country.capitalize()}-{_type.capitalize()}-{reso}', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')
    
    # Save the plot
    os.makedirs(_path, exist_ok=True)
    plt.savefig(os.path.join(_path, f'chronos_{country}_{reso}_{_type}.png'), bbox_inches='tight')
    plt.close()
