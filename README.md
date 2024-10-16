### Developed by: Nivetha A
### Register no.: 212222230101
### Date:

# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 7.5]
train = pd.read_csv('Ex4.csv')
train['date'] = pd.to_datetime(train['date'], format='%d-%m-%Y')
train['Year'] = train['date'].dt.year
train['Values'] = train['open'].values.round().astype(int)
train.head()
ar1 = train['Year'].values.reshape(-1, 1)
ma1 = train['Values'].values.reshape(-1, 1)
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 50])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```
### OUTPUT:
#### SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/user-attachments/assets/b514b871-542c-43d2-a695-a5df259140c9)


#### Partial Autocorrelation
![image](https://github.com/user-attachments/assets/12cda902-32d3-4d68-a214-85776718e5b3)

#### Autocorrelation

![image](https://github.com/user-attachments/assets/5d883a7f-717e-4f35-87e9-75888b2fcd5e)


#### SIMULATED ARMA(2,2) PROCESS:

![image](https://github.com/user-attachments/assets/496858f3-fe9d-498b-afce-96d71a7414f8)

#### Partial Autocorrelation

![image](https://github.com/user-attachments/assets/995f82ce-df1b-454a-9d67-8fb0506ab501)


#### Autocorrelation
![image](https://github.com/user-attachments/assets/8e0594f2-5838-4bb3-a835-95ca7cbec939)

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
