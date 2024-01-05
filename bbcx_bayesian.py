import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from scipy.stats import beta, norm
import base64
import pandas as pd
import io
import pymc3 as pm
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
import arviz as az
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import scipy.optimize  # Import scipy.optimize
import plotly.figure_factory as ff

app = dash.Dash(__name__, suppress_callback_exceptions=True)


# Function to encode image to base64
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded_image}"

# Path to your image file
image_path = '/Users/HopkiFW1/Desktop/fh_bayesian/bbc_studios.png'

# Encode the image to base64
encoded_img = encode_image(image_path)



# Function for Bayesian A/B testing


def calculate_bayesian_sample_size_and_days(min_detectable_effect, baseline_conversion, confidence_level=0.95, power=0.8, daily_traffic=1000, groups_in_test=1):
    # Calculate the required sample size for the specified MDE, confidence level, and power
    adjusted_min_detectable_effect = min_detectable_effect / groups_in_test
    
    with pm.Model() as model:
        # Define prior distributions for conversion rates
        p_baseline = pm.Beta('p_baseline', alpha=1, beta=1)
        p_variant = pm.Beta('p_variant', alpha=1, beta=1)
        
        obs_baseline = pm.Binomial('obs_baseline', n=1, p=p_baseline, observed=baseline_conversion)
        obs_variant = pm.Binomial('obs_variant', n=1, p=p_variant, observed=baseline_conversion + adjusted_min_detectable_effect)
        
        sample_size = pm.Deterministic('sample_size', (2 * (p_baseline - p_variant) / adjusted_min_detectable_effect) ** 2)
    
    with model:
        trace = pm.sample(5000, tune=5000, cores=1)
    
    bayesian_sample_size = int(trace['sample_size'].mean())
    
    days_needed = bayesian_sample_size / daily_traffic
    
    return bayesian_sample_size, days_needed









# Function for Bayesian sample size calculation with continuous data
def calculate_bayesian_sample_size_continuous(min_detectable_effect, baseline_mean, baseline_stddev, confidence_level=0.95, power=0.9, daily_traffic=1000, groups_in_test=1):
    # Define a PyMC3 model
    with pm.Model() as model:
        # Define prior distributions for baseline and variant means
        mu_baseline = pm.Normal('mu_baseline', mu=baseline_mean, sd=baseline_stddev)
        mu_variant = pm.Normal('mu_variant', mu=baseline_mean + min_detectable_effect, sd=baseline_stddev)
        
        # Simulate data for baseline and variant
        data_baseline = pm.Normal('data_baseline', mu=mu_baseline, sd=baseline_stddev)
        data_variant = pm.Normal('data_variant', mu=mu_variant, sd=baseline_stddev)
        
        # Calculate the required sample size for the specified MDE, confidence level, and power
        ppf_alpha = norm.ppf(1 - (1 - confidence_level) / 2)
        ppf_power = norm.ppf(power)
        
        sample_size = (
            (ppf_alpha + ppf_power) ** 2 /
            (min_detectable_effect ** 2)
        )
    
    # Perform MCMC sampling
    with model:
        trace = pm.sample(10000, tune=10000, cores=10)
    
    # Extract the sample size from the trace
    bayesian_sample_size = int(sample_size)
    
    # Calculate the net sample size required for all groups
    net_sample_size = bayesian_sample_size * groups_in_test
    
    # Calculate the number of days needed
    days_needed = net_sample_size / daily_traffic
    
    return net_sample_size, days_needed














def bayesian_ab_test(num_users_control, num_events_control, num_users_variant, num_events_variant):
    # Prior parameters (Beta distribution)
    alpha_prior = 1  # Prior shape parameter
    beta_prior = 1   # Prior scale parameter

    # Calculate posterior parameters for control and variant groups
    alpha_control = alpha_prior + num_events_control
    beta_control = beta_prior + num_users_control - num_events_control

    alpha_variant = alpha_prior + num_events_variant
    beta_variant = beta_prior + num_users_variant - num_events_variant

    # Generate posterior distributions
    posterior_control = beta(alpha_control, beta_control)
    posterior_variant = beta(alpha_variant, beta_variant)

    # Calculate probabilities of variant being better
    p_variant_better = posterior_control.cdf(posterior_variant.mean())

    # Calculate conversion rates for control and variant groups
    conversion_rate_control = num_events_control / num_users_control
    conversion_rate_variant = num_events_variant / num_users_variant

    # Calculate uplift
    uplift = 100 * ((conversion_rate_variant - conversion_rate_control)/conversion_rate_control)

    # Generate KDE plot data for Control and Variant
    x = np.linspace(0, 1, 100)
    kde_control = posterior_control.pdf(x)
    kde_variant = posterior_variant.pdf(x)

    # Define subjective interpretation bands
    result_bands = {
        (0, 0.1): 'strong negative',
        (0.1, 0.2): 'moderate negative',
        (0.2, 0.5): 'weak negative',
        (0.5, 0.8): 'weak positive',
        (0.8, 0.9): 'moderate positive',
        (0.9, 1): 'strong positive'
    }

    # Get the subjective interpretation
    interpretation = ''
    for band, label in result_bands.items():
        if band[0] <= p_variant_better < band[1]:
            interpretation = label
            break

    return {
        'probability_to_be_better': round((p_variant_better * 100),2),
        'conversion_rate_control': round((conversion_rate_control * 100),2),
        'conversion_rate_variant': round((conversion_rate_variant * 100),2),
        'uplift': round((uplift),2),
        'interpretation': f"The probability that the Variant group is better than Control is {round((p_variant_better * 100),2)}%. "
                          f"Uplift: {round((uplift),2)}%. Result interpretation: {interpretation}.",
        'kde_control': kde_control,
        'kde_variant': kde_variant,
        'x_values': x
    }














# Function for Bayesian estimation with continuous data
# Function for continuous Bayesian estimation
def run_continuous_bayesian_estimation(df):
    y1 = df[df['group'] == 'control']['value'].values
    y2 = df[df['group'] == 'variant']['value'].values

    with pm.Model() as model:
        μ_m = df['value'].mean()
        μ_s = df['value'].std() * 2

        group1_mean = pm.Normal("group1_mean", mu=μ_m, sigma=μ_s)
        group2_mean = pm.Normal("group2_mean", mu=μ_m, sigma=μ_s)

        σ_low = 1
        σ_high = 10
        group1_std = pm.Uniform("group1_std", lower=σ_low, upper=σ_high)
        group2_std = pm.Uniform("group2_std", lower=σ_low, upper=σ_high)

        ν = pm.Exponential("ν_minus_one", 1 / 29.0) + 1

        λ1 = group1_std ** -2
        λ2 = group2_std ** -2

        group1 = pm.StudentT("control", nu=ν, mu=group1_mean, lam=λ1, observed=y1)
        group2 = pm.StudentT("variant", nu=ν, mu=group2_mean, lam=λ2, observed=y2)

        trace = pm.sample(5000, return_inferencedata=True)

    az_summary = az.summary(trace, var_names=["group1_mean", "group2_mean"])
    modeled_mean_control = az_summary.loc['group1_mean', 'mean']
    modeled_mean_variant = az_summary.loc['group2_mean', 'mean']

    # Calculate the difference of means manually
    diff_means = trace.posterior["group2_mean"].values - trace.posterior["group1_mean"].values
    uplift = round((100 * ((modeled_mean_variant - modeled_mean_control) / modeled_mean_control)),2)

    prob_group_better = round((diff_means > 0).mean(),2)

    result_text = html.Table(
        # Body
        [html.Tr([html.Td(f"Modeled mean of Control: {modeled_mean_control:.4f}")]),
         html.Tr([html.Td(f"Modeled mean of Variant: {modeled_mean_variant:.4f}")]),
         html.Tr([html.Td(f"Relative uplift from Control to Variant: {uplift:.4f}")]),
         html.Tr([html.Td(f"Probability Variant is better than Control: {prob_group_better:.4f}")])]
    )

    # Displaying the trace as a table
    trace_df = pm.summary(trace).round(4)
    result_table = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in trace_df.columns])] +
        # Body
        [html.Tr([html.Td(trace_df.iloc[i][col]) for col in trace_df.columns]) for i in range(len(trace_df))]
    )

    return result_text, result_table



# Define common style constants
common_style = {
    'textAlign': 'center',
    'color': '#4B0082',  # Pale Purple (BBC Studios color)
    'fontFamily': 'Arial, sans-serif'
}

# Define layout for the first tab
tab1_layout = html.Div([
    html.Img(src=encoded_img, style={'display': 'block', 'margin': '0 auto'}),
    html.H1("BBCX - Bayesian Evaluator", style=common_style),
    
    html.Div([
        html.Label("Control Group", style={'color': '#333', 'fontWeight': 'bold'}),  # Dark Grey
        dcc.Input(id='num_users_control', type='number', placeholder='Number of Users'),
        dcc.Input(id='num_events_control', type='number', placeholder='Number of Events')
    ], style={'margin': '20px auto', 'width': '50%'}),
    
    html.Div([
        html.Label("Variant Group", style={'color': '#333', 'fontWeight': 'bold'}),  # Dark Grey
        dcc.Input(id='num_users_variant', type='number', placeholder='Number of Users'),
        dcc.Input(id='num_events_variant', type='number', placeholder='Number of Events')
    ], style={'margin': '20px auto', 'width': '50%'}),

    html.Button('Run Test', id='run_test', style={'margin': '20px auto', 'width': '200px', 'display': 'block'}),

    html.Div(id='result', style={'color': '#4B0082', 'fontSize': '18px'}),  # Pale Purple

    html.Div([
        html.Hr(style={'borderColor': '#4B0082'}),  # Pale Purple border
        html.P("The 'Probability to be Better' metric is not a strictly mathematical probability. "
               "It serves as a guideline to assist users in decision-making.", style={'color': '#333'}),
        html.P("Result bands interpretation:", style={'fontWeight': 'bold', 'color': '#333'}),  # Dark Grey
        html.P("- 0 to 0.1 indicates a strong negative outcome", style={'color': '#4B0082'}),
        html.P("- 0.1 to 0.2 suggests a moderate negative outcome", style={'color': '#4B0082'}),
        html.P("- 0.2 to 0.5 indicates a weak negative outcome", style={'color': '#4B0082'}),
        html.P("- 0.5 is considered neutral", style={'color': '#4B0082'}),
        html.P("- 0.5 to 0.8 suggests a weak positive outcome", style={'color': '#4B0082'}),
        html.P("- 0.8 to 0.9 indicates a moderate positive outcome", style={'color': '#4B0082'}),
        html.P("- 0.9 to 1 suggests a strong positive outcome.", style={'color': '#4B0082'}),
        html.P("Terms such as weak, moderate, and strong denote the level of confidence users should place in the result. "
               "However, it's essential to note that false positives can potentially occur.", style={'color': '#333'}),
        html.P("Starting Point: We begin with an initial belief (prior distribution) about the success probability. It's like saying, 'I think there's a 50% chance this drug will work.'", style={'color': '#333'}),
        html.P("Random Walk: MCMC performs a random walk through possible success probabilities. It explores different values of success probability, sometimes going higher, sometimes lower. Think of it as exploring the effectiveness of the drug at various levels.", style={'color': '#333'}),
        html.P("Evaluating Likelihood: At each step of the random walk, MCMC checks how well the observed data matches the success probability being considered. If the success probability is close to what we observed, that step is more likely to be accepted.", style={'color': '#333'}),
        html.P("Updating Beliefs: MCMC uses accepted steps to build a new probability distribution called the posterior distribution. This new distribution represents our updated belief about the success probability based on the data.", style={'color': '#333'}),
        html.P("Repeat: MCMC keeps taking steps, evaluating the fit with the data, and updating the beliefs. It does this thousands of times until it converges to a stable posterior distribution.", style={'color': '#333'}),
    html.P("Inference: The final stable posterior distribution tells us the most likely values for the success probability based on the observed data. We can use this to make informed decisions about whether the drug is effective.", style={'color': '#333'}),
    html.P("The 'Probability to be Better' metric is not a strictly mathematical probability. It serves as a guideline to assist users in decision-making.", style={'color': '#333'})
    ], style={'padding': '20px', 'backgroundColor': '#F0F0F0'})  # Light Grey background
])





# Define layout for the second tab
tab2_layout = html.Div([
    html.Img(src=encoded_img, style={'display': 'block', 'margin': '0 auto'}),
    html.H1("BBCX - Bayesian Evaluator", style=common_style),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    
    html.P("This app uses Bayesian estimation and probability distributions to make inference about group differences. It echoes Kruschke's insights, shedding light on the limitations of traditional statistical tests by showcasing the flaws in comparing groups. Rather than merely evaluating differences, it employs diverse probability distributions like Student-t to comprehensively estimate group disparities while factoring in uncertainties—a substantial leap from conventional methods.", style={'color': '#333'}),
    
    html.P("PyMC3, at the core of this application, facilitates the establishment of priors, model fitting, and post-analysis visualization, aligning with the nuanced Bayesian Estimation approach. It steers away from the conventional norm by incorporating probability models, notably in setting up group means, standard deviations, and degrees of freedom parameters. While not explicitly diving into drug trials, this app brings forth the relevance of Bayesian Estimation's prowess in enhancing A/B testing methodologies.", style={'color': '#333'}),
    
    html.P("By adopting varied distribution models and leveraging PyMC3, it amplifies the depth and reliability of insights derived from A/B tests. Its emphasis on Bayesian Estimation's versatility and the application of different distribution models establishes it as an invaluable tool for analysts and researchers aiming for comprehensive and reliable test result interpretations.", style={'color': '#333'}),
    
    html.Div(id='output-data-upload'),
    html.Div(id='bayesian-results'),
    
html.Div([
    html.Hr(style={'borderColor': '#4B0082'}),  # Pale Purple border
    html.P("Result bands interpretation:", style={'fontWeight': 'bold', 'color': '#333'}),  # Dark Grey
    html.P("- 0 to 0.1 indicates a strong negative outcome", style={'color': '#4B0082'}),
    html.P("- 0.1 to 0.2 suggests a moderate negative outcome", style={'color': '#4B0082'}),
    html.P("- 0.2 to 0.5 indicates a weak negative outcome", style={'color': '#4B0082'}),
    html.P("- 0.5 is considered neutral", style={'color': '#4B0082'}),
    html.P("- 0.5 to 0.8 suggests a weak positive outcome", style={'color': '#4B0082'}),
    html.P("- 0.8 to 0.9 indicates a moderate positive outcome", style={'color': '#4B0082'}),
    html.P("- 0.9 to 1 suggests a strong positive outcome.", style={'color': '#4B0082'}),
    html.P("Terms such as weak, moderate, and strong denote the level of confidence users should place in the result. However, it's essential to note that false positives can potentially occur.", style={'color': '#333'}),
    html.P("Starting Point: We begin with an initial belief (prior distribution) about the success probability. It's like saying, 'I think there's a 50% chance this drug will work.'", style={'color': '#333'}),
    html.P("Random Walk: MCMC performs a random walk through possible success probabilities. It explores different values of success probability, sometimes going higher, sometimes lower. Think of it as exploring the effectiveness of the drug at various levels.", style={'color': '#333'}),
    html.P("Evaluating Likelihood: At each step of the random walk, MCMC checks how well the observed data matches the success probability being considered. If the success probability is close to what we observed, that step is more likely to be accepted.", style={'color': '#333'}),
    html.P("Updating Beliefs: MCMC uses accepted steps to build a new probability distribution called the posterior distribution. This new distribution represents our updated belief about the success probability based on the data.", style={'color': '#333'}),
    html.P("Repeat: MCMC keeps taking steps, evaluating the fit with the data, and updating the beliefs. It does this thousands of times until it converges to a stable posterior distribution.", style={'color': '#333'}),
    html.P("Inference: The final stable posterior distribution tells us the most likely values for the success probability based on the observed data. We can use this to make informed decisions about whether the drug is effective.", style={'color': '#333'}),
    html.P("The 'Probability to be Better' metric is not a strictly mathematical probability. It serves as a guideline to assist users in decision-making.", style={'color': '#333'}),
],
style={'margin': '20px auto', 'width': '90%'})
  # Light Grey background
])









# Define layout for the third tab (Tab 3 - Binomial Sample Size Calculator)
tab3_layout = html.Div([
    html.Img(src=encoded_img, style={'display': 'block', 'margin': '0 auto'}),
    html.H1("Binomial Sample Size Calculator", style=common_style),

    html.Div([
        html.P("In a Bayesian context, sample size calculation is essential for designing experiments that allow you to make informed decisions based on observed data. The calculator uses your inputs to estimate the sample size required to detect a minimum detectable effect (MDE) with a specified confidence level and power.", style={'color': '#333'}),
        html.P("To use the calculator, input the following values:", style={'color': '#333'}),
        html.Label("Minimum Detectable Effect (MDE):", style={'fontWeight': 'bold'}),
        dcc.Input(id="mde_input", type="number", value=0.01),
    ], style={'margin': '20px auto', 'width': '90%'}),

    html.Div([
        html.P("Baseline Conversion Rate: This is the expected conversion rate in the control group (before any changes or interventions). Enter a decimal value, e.g., 0.033 for 3.3%.", style={'color': '#333'}),
        html.Label("Baseline Conversion Rate:", style={'fontWeight': 'bold'}),
        dcc.Input(id="baseline_input", type="number", value=0.033),
    ], style={'margin': '20px auto', 'width': '90%'}),

    html.Div([
        html.P("Daily Traffic: This is the number of users or events you expect to encounter in a day. It helps estimate the time required to collect the desired sample size. Enter a whole number, e.g., 1000.", style={'color': '#333'}),
        html.Label("Daily Traffic:", style={'fontWeight': 'bold'}),
        dcc.Input(id="daily_traffic_input", type="number", value=1000),
    ], style={'margin': '20px auto', 'width': '90%'}),

    html.Div([
        html.P("Number of Groups in Test: Specify the number of groups or variations in your experiment. If you're testing a single variation against a control group, enter 1. If you have multiple variations, enter the number of variations.", style={'color': '#333'}),
        html.Label("Number of Groups in Test:", style={'fontWeight': 'bold'}),
        dcc.Input(id="groups_input", type="number", value=1),
    ], style={'margin': '20px auto', 'width': '90%'}),

    html.Button("Calculate Sample Size", id="calculate_sample_button", style={'margin': '40px auto', 'width': '300px'}),

    html.Div(id="sample_size_result", style={'fontSize': '18px'}),
    
    html.Div([
        html.H3("Understanding Sample Size Analysis in a Bayesian Context", style={'color': '#222'}),
        html.P("Sample size analysis in a Bayesian context is a critical step in experimental design. Bayesian statistics allows us to incorporate prior information and update our beliefs based on observed data. In this context, sample size calculation helps us determine the amount of data needed to achieve our research goals.", style={'color': '#333'}),
        html.P("1. The Problem: Estimating Sample Size - Let's say you want to figure out how many data points (samples) you need to collect for an experiment, like a scientific study or an A/B test. You want to ensure that your study is statistically meaningful and can provide reliable results.", style={'color': '#333'}),
        html.P("2. Complex Probability Distribution: Uncertainty in Sample Size - The number of samples you need depends on various factors, like the desired level of confidence, the expected effect size, and other statistical parameters. These factors create a complex probability distribution representing the uncertainty in the required sample size. It's like a mountain with peaks and valleys, where each point on the mountain represents a possible sample size.", style={'color': '#333'}),
        html.P("3. MCMC as an Explorer: Randomly Exploring the Mountain - MCMC acts as an explorer in this scenario. It starts with an initial guess for the sample size, and then it takes random steps to explore different values of the sample size. These steps are guided by the complex probability distribution you're trying to understand. Sometimes it takes steps that lead to larger sample sizes, and sometimes to smaller ones, all in a random fashion.", style={'color': '#333'}),
        html.P("4. Collecting Information: Building a Map of Sample Sizes - As MCMC takes more steps, it keeps track of the sample sizes it has explored and how often it visits each value. This information is used to create a map of the probability distribution of sample sizes. Just like before, the more steps it takes, the more.", style={'color': '#333'}),
        html.P("5. Reaching Equilibrium: Finding the Most Suitable Sample Size - With enough steps, MCMC tends to spend more time around sample sizes that are more likely to be appropriate for your study. It's like being drawn to the peaks (optimal sample sizes) in the distribution. This is crucial because you want your study to be statistically sound and efficient..", style={'color': '#333'}),
        html.P("6. Estimating Sample Size: Calculating the Required Sample Size - By looking at the areas of the distribution where MCMC has spent the most time, you can estimate the required sample size for your experiment. This estimated sample size is statistically meaningful and based on the uncertainty in your problem..", style={'color': '#333'})
    ], style={'margin': '20px auto', 'width': '90%'}),
])







# Define layout for the fourth tab (Tab 4 - Bayesian Sample Size Calculator with Continuous Data)
tab4_layout = html.Div([
    html.Img(src=encoded_img, style={'display': 'block', 'margin': '0 auto'}),
    html.H1("Continuous Sample Size Calculator", style=common_style),
    html.Div([
        html.P("In a Bayesian context, sample size calculation is essential for designing experiments that allow you to make informed decisions based on observed data. The calculator uses your inputs to estimate the sample size required to detect a minimum detectable effect (MDE) with a specified confidence level and power.", style={'color': '#333'}),
        html.P("To use the calculator, input the following values:", style={'color': '#333'}),
        
        html.Label("Minimum Detectable Effect (MDE):", style={'fontWeight': 'bold'}),
        dcc.Input(id='mde_input_continuous', type='number', placeholder='Enter MDE'),
        
        html.P("Minimum Detectable Effect (MDE): This is the smallest effect size that you want to be able to detect in your experiment. It represents the practical significance of your findings. Enter a decimal value, e.g., 0.01 for 1%.", style={'color': '#333'}),
        
        html.Label("Baseline Mean:", style={'fontWeight': 'bold'}),
        dcc.Input(id='baseline_mean_input', type='number', placeholder='Enter Baseline Mean'),
        
        html.P("Baseline Mean: This is the expected mean or average value in the control group (before any changes or interventions). Enter a decimal value.", style={'color': '#333'}),
        
        html.Label("Baseline Standard Deviation:", style={'fontWeight': 'bold'}),
        dcc.Input(id='baseline_stddev_input', type='number', placeholder='Enter Baseline Std Dev'),
        
        html.P("Baseline Standard Deviation: This is the expected standard deviation of the data in the control group. Enter a decimal value.", style={'color': '#333'}),
        
        html.Label("Daily Traffic:", style={'fontWeight': 'bold'}),
        dcc.Input(id='daily_traffic_input_continuous', type='number', placeholder='Enter Daily Traffic'),
        
        html.P("Daily Traffic: This is the number of observations or data points you expect to collect in a day. It helps estimate the time required to collect the desired sample size. Enter a whole number.", style={'color': '#333'}),
        
        html.Label("Groups in Test:", style={'fontWeight': 'bold'}),
        dcc.Input(id='groups_input_continuous', type='number', placeholder='Enter Groups in Test'),
        
        html.P("Groups in Test: Specify the number of groups or variations in your experiment. If you're testing a single variation against a control group, enter 1. If you have multiple variations, enter the number of variations.", style={'color': '#333'}),
        
        html.Button('Calculate Sample Size', id='calculate_sample_button_continuous', style={'margin': '20px auto', 'width': '200px', 'display': 'block'}),
        
        html.Div(id='sample_size_result_continuous', style={'fontSize': '18px'}),
    ], style={'margin': '20px auto', 'width': '90%'}),
    
    html.Div([
        html.H3("Understanding Sample Size Analysis in a Bayesian Context", style={'color': '#222'}),
        html.P("Sample size analysis in a Bayesian context is a critical step in experimental design. Bayesian statistics allows us to incorporate prior information and update our beliefs based on observed data. In this context, sample size calculation helps us determine the amount of data needed to achieve our research goals.", style={'color': '#333'}),
        html.P("1. The Problem: Estimating Sample Size - Let's say you want to figure out how many data points (samples) you need to collect for an experiment, like a scientific study or an A/B test. You want to ensure that your study is statistically meaningful and can provide reliable results.", style={'color': '#333'}),
        html.P("2. Complex Probability Distribution: Uncertainty in Sample Size - The number of samples you need depends on various factors, like the desired level of confidence, the expected effect size, and other statistical parameters. These factors create a complex probability distribution representing the uncertainty in the required sample size. It's like a mountain with peaks and valleys, where each point on the mountain represents a possible sample size.", style={'color': '#333'}),
        html.P("3. MCMC as an Explorer: Randomly Exploring the Mountain - MCMC acts as an explorer in this scenario. It starts with an initial guess for the sample size, and then it takes random steps to explore different values of the sample size. These steps are guided by the complex probability distribution you're trying to understand. Sometimes it takes steps that lead to larger sample sizes, and sometimes to smaller ones, all in a random fashion.", style={'color': '#333'}),
        html.P("4. Collecting Information: Building a Map of Sample Sizes - As MCMC takes more steps, it keeps track of the sample sizes it has explored and how often it visits each value. This information is used to create a map of the probability distribution of sample sizes. Just like before, the more steps it takes, the more.", style={'color': '#333'}),
        html.P("5. Reaching Equilibrium: Finding the Most Suitable Sample Size - With enough steps, MCMC tends to spend more time around sample sizes that are more likely to be appropriate for your study. It's like being drawn to the peaks (optimal sample sizes) in the distribution. This is crucial because you want your study to be statistically sound and efficient..", style={'color': '#333'}),
        html.P("6. Estimating Sample Size: Calculating the Required Sample Size - By looking at the areas of the distribution where MCMC has spent the most time, you can estimate the required sample size for your experiment. This estimated sample size is statistically meaningful and based on the uncertainty in your problem..", style={'color': '#333'})
    ], style={'margin': '20px auto', 'width': '90%'}),
])














@app.callback(
    Output('bayesian-results', 'children'),  # Update this element
    Input('upload-data', 'contents')
, allow_duplicate=True)
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        def generate_kde_plot(df):
            results = []  # Create an empty list to store sampled means
            random_state = np.arange(0, 1500)  # Random seeds for reproducibility
        
            # Generate sampled means
            for i in range(1500):
                sample = df.sample(frac=0.5, replace=True, random_state=random_state[i]).groupby(by='group')['value'].mean()
                results.append(sample)
            dist_samples = pd.DataFrame(results)
            
            # Melt the data for visualization
            df = dist_samples.melt(var_name='group', value_name='value')
        
            # Create KDE plot
            fig = ff.create_distplot(
                [df[df['group'] == 'control']['value'], df[df['group'] == 'variant']['value']],
                group_labels=['Control', 'Variant'],
                show_hist=False,
                curve_type='kde'
            )
        
            # Update layout
            fig.update_layout(
                title='KDE Plot for Control vs Variant',
                xaxis_title='Value',
                yaxis_title='Density'
            )
        
            return dcc.Graph(figure=fig)
        # Call function to generate KDE plot
        kde_plot = generate_kde_plot(df)

        def run_bayesian_estimation(df):
            y1 = df[df['group'] == 'control']['value'].values
            y2 = df[df['group'] == 'variant']['value'].values
        
            with pm.Model() as model:
                μ_m = df['value'].mean()
                μ_s = df['value'].std() * 2
        
                group1_mean = pm.Normal("group1_mean", mu=μ_m, sigma=μ_s)
                group2_mean = pm.Normal("group2_mean", mu=μ_m, sigma=μ_s)
        
                σ_low = 1
                σ_high = 10
                group1_std = pm.Uniform("group1_std", lower=σ_low, upper=σ_high)
                group2_std = pm.Uniform("group2_std", lower=σ_low, upper=σ_high)
        
                ν = pm.Exponential("ν_minus_one", 1 / 29.0) + 1
        
                λ1 = group1_std ** -2
                λ2 = group2_std ** -2
        
                group1 = pm.StudentT("control", nu=ν, mu=group1_mean, lam=λ1, observed=y1)
                group2 = pm.StudentT("variant", nu=ν, mu=group2_mean, lam=λ2, observed=y2)
        
                trace = pm.sample(5000, return_inferencedata=True)
        
            az_summary = az.summary(trace, var_names=["group1_mean", "group2_mean"])
            modeled_mean_control = az_summary.loc['group1_mean', 'mean']
            modeled_mean_variant = az_summary.loc['group2_mean', 'mean']
        
            # Calculate the difference of means manually
            diff_means = trace.posterior["group2_mean"].values - trace.posterior["group1_mean"].values
            uplift_cont = 100 * ((modeled_mean_variant - modeled_mean_control) / modeled_mean_control)
        
            prob_group_better_cont = round((diff_means > 0).mean(),2)
            
            result_text = html.Table(
                # Body
                [html.Tr([html.Td(f"Modeled mean of Control: {modeled_mean_control:.4f}")]),
                 html.Tr([html.Td(f"Modeled mean of Variant: {modeled_mean_variant:.4f}")]),
                 html.Tr([html.Td(f"Relative uplift from Control to Variant: {round(uplift_cont,2)}%")]),
                 html.Tr([html.Td(f"Probability Variant is better than Control: {100*prob_group_better_cont}%")])]
            )

            # Displaying the trace as a table
            trace_df = pm.summary(trace).round(4)
            result_table = html.Table(
                # Header
                [html.Tr([html.Th(col) for col in trace_df.columns])] +
                # Body
                [html.Tr([html.Td(trace_df.iloc[i][col]) for col in trace_df.columns]) for i in range(len(trace_df))]
            )
        
            return result_text, result_table

        # Call Bayesian estimation function
        text_output, table_output = run_bayesian_estimation(df)

        # Return the KDE plot along with other elements
        return html.Div([kde_plot, html.Div([html.Div(html.P(text_output)), html.Div(table_output)])])

    return None







 



# Callback for the binomial AB Test tab
@app.callback(Output('result', 'children'),  # Change the Output ID accordingly
              Input('run_test', 'n_clicks'),
              [dash.dependencies.State('num_users_control', 'value'),
               dash.dependencies.State('num_events_control', 'value'),
               dash.dependencies.State('num_users_variant', 'value'),
               dash.dependencies.State('num_events_variant', 'value')])
def run_ab_test(n_clicks, num_users_control, num_events_control, num_users_variant, num_events_variant):
    if None not in [num_users_control, num_events_control, num_users_variant, num_events_variant]:
        # Call the Bayesian A/B testing function
        result = bayesian_ab_test(num_users_control, num_events_control, num_users_variant, num_events_variant)

        # Merge KDE plots for Control and Variant
        trace_control = go.Scatter(x=result['x_values'], y=result['kde_control'], mode='lines', name='Control')
        trace_variant = go.Scatter(x=result['x_values'], y=result['kde_variant'], mode='lines', name='Variant')
        merged_plot = [trace_control, trace_variant]

        # Create a Dash-friendly output
        return [
            html.H2("Result", style={'color': '#222', 'fontFamily': 'Arial, sans-serif'}),
            html.P(f"Probability of Variant group being better than Control: {result['probability_to_be_better']}%", style={'color': '#222'}),
            html.P(f"Conversion Rate for Control Group: {result['conversion_rate_control']}%", style={'color': '#222'}),
            html.P(f"Conversion Rate for Variant Group: {result['conversion_rate_variant']}%", style={'color': '#222'}),
            html.P(f"Uplift: {result['uplift']}%", style={'color': '#222'}),
            html.P(result['interpretation'], style={'color': '#222'}),
            dcc.Graph(figure={'data': merged_plot})
        ]






app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Binomial A/B Test', value='tab1', style=common_style),
        dcc.Tab(label='Continuous A/B Test', value='tab2', style=common_style),
        dcc.Tab(label='Binomial Sample Size Calculator', value='tab3', style=common_style),
        dcc.Tab(label='Continuous Sample Size Calculator', value='tab4', style=common_style),
    ]),
    html.Div(id='tabs-content')
])





# Update the callback function to match the IDs in the layout
@app.callback(Output('sample_size_result', 'children'),  # Update the Output ID
              Input('calculate_sample_button', 'n_clicks'),
              [dash.dependencies.State('mde_input', 'value'),
               dash.dependencies.State('baseline_input', 'value'),
               dash.dependencies.State('daily_traffic_input', 'value'),
               dash.dependencies.State('groups_input', 'value')])
def run_sample_size_calculator(n_clicks, min_detectable_effect, baseline_conversion, daily_traffic, groups_in_test):
    if n_clicks is not None:
        if None not in [min_detectable_effect, baseline_conversion, daily_traffic, groups_in_test]:
            bayesian_sample_size, days_needed = calculate_bayesian_sample_size_and_days(min_detectable_effect, baseline_conversion, daily_traffic=daily_traffic, groups_in_test=groups_in_test)

            return [
                html.H2("Sample Size Calculation Result", style={'color': '#222', 'fontFamily': 'Arial, sans-serif'}),
                html.P(f"Required Sample Size: {bayesian_sample_size}", style={'color': '#222'}),
                html.P(f"Estimated Days Needed: {days_needed}", style={'color': '#222'}),
            ]





@app.callback(Output('sample_size_result_continuous', 'children'),
              Input('calculate_sample_button_continuous', 'n_clicks'),
              [dash.dependencies.State('mde_input_continuous', 'value'),
               dash.dependencies.State('baseline_mean_input', 'value'),
               dash.dependencies.State('baseline_stddev_input', 'value'),
               dash.dependencies.State('daily_traffic_input_continuous', 'value'),
               dash.dependencies.State('groups_input_continuous', 'value')])
def run_sample_size_calculator_continuous(n_clicks, min_detectable_effect_continuous, baseline_mean, baseline_stddev, daily_traffic, groups_in_test):
    if n_clicks is not None:
        if None not in [min_detectable_effect_continuous, baseline_mean, baseline_stddev, daily_traffic, groups_in_test]:
            bayesian_sample_size_continuous, days_needed_continuous = calculate_bayesian_sample_size_continuous(
                min_detectable_effect_continuous, baseline_mean, baseline_stddev, daily_traffic=daily_traffic, groups_in_test=groups_in_test)

            return [
                html.H2("Sample Size Calculation Result (Continuous)", style={'color': '#222', 'fontFamily': 'Arial, sans-serif'}),
                html.P(f"Required Sample Size: {bayesian_sample_size_continuous}", style={'color': '#222'}),
                html.P(f"Estimated Days Needed: {days_needed_continuous}", style={'color': '#222'}),
            ]










# Callback to switch tabs
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab1':
        return tab1_layout
    elif tab == 'tab2':
        return tab2_layout
    elif tab == 'tab3':  # New tab
        return tab3_layout
    elif tab == 'tab4':  # New tab
        return tab4_layout



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)