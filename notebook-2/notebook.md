# Predicting Medical Insurance Cost using Ensemble Methods

Data science and machine learning can be apllied in many areas and multiple domains. Today, I will aplly my humble knowledge of ML in the medical field, by giving predictions for insurance prices, which can be useful in various situations.  
The aim of this project is to predict the cost of meeical insurance using data of different people. It is a simple project that helpd me practice the basics of Machine Learning that I learned online.  
The strategy for the project is following:  

- Import the data and various libraries
- A Data exploration phase
- Data preprocessing and creation of sets
- Choice of a model and testing the model

## Import the data and libraries:


```python
import sklearn
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
```

Even if it isn't necessary is our case, we can create a funcion to save the plots or images on our computer, which can be useful for certain projects.


```python
project_root_dir = "."
images_path = os.path.join(project_root_dir, "images")
os.makedirs(images_path, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

Next we can import our dataset which is in a CSV format. We use Pandas to open the CSV


```python
housing_path = os.path.join("datasets", "insurance.csv")

def load_insurance_data(housing_path=housing_path):
    csv_path = housing_path
    return pd.read_csv(csv_path)

insurance = load_insurance_data()
print(insurance.head())
```

       age     sex     bmi  children smoker     region      charges
    0   19  female  27.900         0    yes  southwest  16884.92400
    1   18    male  33.770         1     no  southeast   1725.55230
    2   28    male  33.000         3     no  southeast   4449.46200
    3   33    male  22.705         0     no  northwest  21984.47061
    4   32    male  28.880         0     no  northwest   3866.85520


This is what our dataset looks like! We can see that it contains some basic data about the patients, and of course the charges, which are the insurance costs.

## Data Exploration phase:

We can use the "describe" method of our dataframe to obtain some additional information. For example, we can see that there is no missing data, or we observe the ranges for different numerical features.


```python
insurance.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>
</div>



We can further explore our dataset with some visualisation tools: the previous "describe" method gives a lot of information, but plotting the data (with histograms) should give us a better idea of the distribution.


```python
%matplotlib inline
import matplotlib.pyplot as plt
insurance.hist(bins=25, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
```


    
![png](output_13_0.png)
    


Fun fact: if we observe the distribution for the charges, it most resembles a normal distribution, which is quite often the case when gathering data from enough people.

## Data Preprocessing and Creation of Training/Test Sets:

For practical resaons we will set a seed in the notebook to have the same results if we run it multiple times.


```python
np.random.seed(42)
```

We then use the **train_test_split** function from sklearn to create sets of data for training the model and then testing it. This is how it's done :


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(insurance, test_size=0.2, random_state=42)
```

Quick look at the test set, to check if rows are randomly selected:


```python
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>764</th>
      <td>45</td>
      <td>female</td>
      <td>25.175</td>
      <td>2</td>
      <td>no</td>
      <td>northeast</td>
      <td>9095.06825</td>
    </tr>
    <tr>
      <th>887</th>
      <td>36</td>
      <td>female</td>
      <td>30.020</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>5272.17580</td>
    </tr>
    <tr>
      <th>890</th>
      <td>64</td>
      <td>female</td>
      <td>26.885</td>
      <td>0</td>
      <td>yes</td>
      <td>northwest</td>
      <td>29330.98315</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>46</td>
      <td>male</td>
      <td>25.745</td>
      <td>3</td>
      <td>no</td>
      <td>northwest</td>
      <td>9301.89355</td>
    </tr>
    <tr>
      <th>259</th>
      <td>19</td>
      <td>male</td>
      <td>31.920</td>
      <td>0</td>
      <td>yes</td>
      <td>northwest</td>
      <td>33750.29180</td>
    </tr>
  </tbody>
</table>
</div>



Now we will work on the training set only, so we make a copy of it before modifications:


```python
insurance = train_set.copy()
```

Let's continue our Data Exploration phase : we can first take a look at how the charges behave with the age of the patients. The reason for this plot is that we want to see if there is a direct "link" between age and medical charges for insurance. Let's check:


```python
insurance.plot(kind="scatter",x="age",y="charges",alpha=0.1)
save_fig("visualization_charges_against_age")
```


    
![png](output_25_0.png)
    


By setting the "alpha" to 0.1, we can modify the transparency of the plot. This is helpful because it is easier to see higher density of points on the graph. Looking at the plot, we can see that there is a strong density at the bottom, and some points at the top, less densely dispatched.

Another technique when exploring your data is to look for correlations between different features. For that, we can use the **corr()** method for a dataframe, by only selecting the numerical features of the dataset.


```python
insurance_num = insurance[["age","bmi","children","charges"]]
corr_matrix = insurance_num.corr()
corr_matrix["charges"].sort_values(ascending=False)
```




    charges     1.000000
    age         0.281721
    bmi         0.197316
    children    0.071885
    Name: charges, dtype: float64



**Comments on the correlation matrix** : we can see that the "age" feature  has the highest correlation with the "charges". So when predicting the amount of charges for a patient, it would be more important to ask for their age rather than the number of children. Also the "BMI" fetaure (Body Mass Index) is also somewhat correlated with charges.

We can also plot a **Scatter Matrix**, which is simply a matrix of scatterplots. This matrix will allow us to visualise the relationship between multiple features at once.


```python
from pandas.plotting import scatter_matrix

scatter_matrix(insurance_num, figsize=(12, 8))
save_fig("scatter_matrix_plot")
```


    
![png](output_31_0.png)
    


__Remark__ : You can also experiment with new parameters by trying new combinations ; in our case the number of numeric attribute is too low so we wil not try to combine them. If the model is not performing good enough we can try to come back here to add other attributes with hiden correlation.

Now it is time to "clean" our data and make it ready to go into the model: we will first drop the aim column ("charges" column).


```python
insurance = train_set.drop("charges", axis=1) 
insurance_num = insurance[["age","bmi","children"]]
insurance_labels = train_set["charges"].copy()
```


```python
sample_incomplete_rows = insurance[insurance.isnull().any(axis=1)].head()
sample_incomplete_rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



__Remark__ : here we can see that the dataset is well constructed as there are no missing values ! 

__Other Remark__ : On a more serious note, we chose this dataset because it was easy to use. In reality the datasets are almost never complete and they require good amout of cleaning.

To handle text and categorical attributes, we can use the **OneHotEncoder** from sklearn: it transforms our data so that our "non-numerical" data becomes numerical!


```python
insurance_cat = insurance[["sex","smoker","region"]]
insurance_cat.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>smoker</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>560</th>
      <td>female</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
    <tr>
      <th>1285</th>
      <td>female</td>
      <td>no</td>
      <td>northeast</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>female</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>969</th>
      <td>female</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>486</th>
      <td>female</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
    <tr>
      <th>170</th>
      <td>male</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>277</th>
      <td>female</td>
      <td>no</td>
      <td>southwest</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>male</td>
      <td>yes</td>
      <td>northeast</td>
    </tr>
    <tr>
      <th>209</th>
      <td>male</td>
      <td>no</td>
      <td>northeast</td>
    </tr>
    <tr>
      <th>947</th>
      <td>male</td>
      <td>yes</td>
      <td>northeast</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
insurance_cat_1hot = cat_encoder.fit_transform(insurance_cat)
insurance_cat_1hot
```




    <1070x8 sparse matrix of type '<class 'numpy.float64'>'
    	with 3210 stored elements in Compressed Sparse Row format>




```python
cat_encoder.categories_
```




    [array(['female', 'male'], dtype=object),
     array(['no', 'yes'], dtype=object),
     array(['northeast', 'northwest', 'southeast', 'southwest'], dtype=object)]



Another technique is to create custom transformers: custom transformers are used to build custom cleaning operations or to combine specific attributes. Here we have too few attributes in the dataset, so we will try to continue without implementing new attributes. For illustration purposes this is what a custom transformer could look like :


```python
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
```

Let's build some pipelines for the transformation of our data :

__Remark__ : The first pipeline (pipeline_num) is used to apply the needed transformations (handling missing values and scaling the values) on the numerical atrributes.
The second pipeline (full_pipeline) is used to combine the transformations on numerical attributes and the transformations on the categorical (or non numerical) attributes.


```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

insurance_num_tr = num_pipeline.fit_transform(insurance_num)
```


```python
insurance_num_tr
```




    array([[ 0.47222651, -1.75652513,  0.73433626],
           [ 0.54331294, -1.03308239, -0.91119211],
           [ 0.8987451 , -0.94368672, -0.91119211],
           ...,
           [ 1.3252637 , -0.89153925, -0.91119211],
           [-0.16755139,  2.82086429,  0.73433626],
           [ 1.1120044 , -0.10932713, -0.91119211]])



__Remark__ : Here the SimpleImputer() transformer is only there for illustration purposes. We already checked that there were no missing values in the dataset. If it was the case we could have used this method.


```python
from sklearn.compose import ColumnTransformer

num_attribs = ["age","bmi","children"]
cat_attribs = list(insurance_cat)

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

insurance_prepared = full_pipeline.fit_transform(insurance)
```


```python
insurance_prepared
```




    array([[ 0.47222651, -1.75652513,  0.73433626, ...,  1.        ,
             0.        ,  0.        ],
           [ 0.54331294, -1.03308239, -0.91119211, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.8987451 , -0.94368672, -0.91119211, ...,  0.        ,
             1.        ,  0.        ],
           ...,
           [ 1.3252637 , -0.89153925, -0.91119211, ...,  0.        ,
             0.        ,  0.        ],
           [-0.16755139,  2.82086429,  0.73433626, ...,  0.        ,
             0.        ,  1.        ],
           [ 1.1120044 , -0.10932713, -0.91119211, ...,  0.        ,
             0.        ,  1.        ]])




```python
insurance_prepared.shape
```




    (1070, 11)



## Choice of a Model and Testing Phase :

Let's first try a linear regression :


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(insurance_prepared, insurance_labels)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



Let's try the full preprocessing pipeline on a few training samples: 


```python
some_data = insurance.iloc[:5]
some_labels = insurance_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
```

    Predictions: [ 6656.  8704.  9216. 10752.  9536.]



```python
print("Labels:", list(some_labels))
```

    Labels: [9193.8385, 8534.6718, 27117.99378, 8596.8278, 12475.3513]



```python
from sklearn.metrics import mean_squared_error

insurance_predictions = lin_reg.predict(insurance_prepared)
lin_mse = mean_squared_error(insurance_labels, insurance_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    6113.551151636188




```python
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(insurance_labels, insurance_predictions)
lin_mae
```




    4282.680959299066



Let's now try a Decision Tree Regressor :


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(insurance_prepared, insurance_labels)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor(random_state=42)</pre></div></div></div></div></div>




```python
insurance_predictions = tree_reg.predict(insurance_prepared)
tree_mse = mean_squared_error(insurance_labels, insurance_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

```




    494.20598375812835



We can try to get a better evaluation using **Cross-Validation** :


```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, insurance_prepared, insurance_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
```

    Scores: [6339.58900717 6578.46280707 7070.94331065 7082.71162807 7035.35755633
     6537.63856176 7436.65224436 7411.85619806 6621.32660141 5569.81488004]
    Mean: 6768.4352794914985
    Standard deviation: 534.1938075533002



```python
lin_scores = cross_val_score(lin_reg, insurance_prepared, insurance_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
```

    Scores: [6088.62821183 6579.03828457 5218.86367097 6042.09621288 5843.70905614
     6177.20119308 7212.64087053 6318.98648379 6191.88536253 5669.09794393]
    Mean: 6134.214729025356
    Standard deviation: 504.9102849483168


Let's now try a Random Forest Regressor : it works by training many Decision Trees on random subsets of the features, then averaging out their predictions. 


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(insurance_prepared, insurance_labels)

insurance_predictions = forest_reg.predict(insurance_prepared)
forest_mse = mean_squared_error(insurance_labels, insurance_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    1900.9799788695125



Let's get the CV Score for the Random Forest Regressor :


```python
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, insurance_prepared, insurance_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

    Scores: [4837.88091145 5321.29233914 3855.38071452 4444.14090314 5283.52272899
     4882.71894641 5656.0522389  5424.90384931 5177.82380257 4687.58891207]
    Mean: 4957.130534650082
    Standard deviation: 506.4164192696564


 Conclusion : the Random Forest Regressor seems to be the best performing model.

##  Bonus Part, Fine-tuning the selected model (Random Forest Regressor) :

We will try to get the best version possible of our model. For that, we can look for the best hyperparameters, which are parameters that are set at the beginning of training and that are not learned. The **GridSearchCV** method allows us to do two things at the same time:

- The GridSearch part : different combinations of parameters are tested in the model to see whiwh one works best.
- The CrossValidation part : to test each combination og hyperparameters, we do not just run the model once; instead we use the CV method by using a number k of folds. We then train the model on k-1 folds and test it one the remaining fold. Of course we reiterate as many times as the number of folds we decided to have.


```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    # (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 5]},
    # (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(insurance_prepared, insurance_labels)
```




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
             param_grid=[{&#x27;max_features&#x27;: [2, 3, 4, 5],
                          &#x27;n_estimators&#x27;: [3, 10, 30]},
                         {&#x27;bootstrap&#x27;: [False], &#x27;max_features&#x27;: [2, 3, 4],
                          &#x27;n_estimators&#x27;: [3, 10]}],
             return_train_score=True, scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
             param_grid=[{&#x27;max_features&#x27;: [2, 3, 4, 5],
                          &#x27;n_estimators&#x27;: [3, 10, 30]},
                         {&#x27;bootstrap&#x27;: [False], &#x27;max_features&#x27;: [2, 3, 4],
                          &#x27;n_estimators&#x27;: [3, 10]}],
             return_train_score=True, scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>




```python
grid_search.best_params_
```




    {'max_features': 5, 'n_estimators': 30}




```python
grid_search.best_estimator_
```




<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_features=5, n_estimators=30, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" checked><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(max_features=5, n_estimators=30, random_state=42)</pre></div></div></div></div></div>




```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    5869.472669568795 {'max_features': 2, 'n_estimators': 3}
    5344.09974595216 {'max_features': 2, 'n_estimators': 10}
    5043.721468813344 {'max_features': 2, 'n_estimators': 30}
    5648.138086537294 {'max_features': 3, 'n_estimators': 3}
    5105.500999588195 {'max_features': 3, 'n_estimators': 10}
    4893.003223784434 {'max_features': 3, 'n_estimators': 30}
    5408.495942571259 {'max_features': 4, 'n_estimators': 3}
    4917.741867902749 {'max_features': 4, 'n_estimators': 10}
    4819.911075212028 {'max_features': 4, 'n_estimators': 30}
    5329.890795940145 {'max_features': 5, 'n_estimators': 3}
    4945.846057937126 {'max_features': 5, 'n_estimators': 10}
    4791.773863128718 {'max_features': 5, 'n_estimators': 30}
    5824.642078297096 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    5360.687013306672 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    5590.669350674112 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    5215.110265030829 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    5596.97740383503 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    5219.18068725983 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}



```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([0.14203938, 0.18778845, 0.02184478, 0.00542714, 0.00411807,
           0.27403738, 0.34359555, 0.00547049, 0.00498177, 0.00657547,
           0.00412152])




```python
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.18778844973966602, 'bmi'),
     (0.14203937513086864, 'age'),
     (0.02184477637216532, 'children'),
     (0.005427140684887419, 'female'),
     (0.004118072604870407, 'male')]



## Last step of the project: Testing the Model !

We are now going to test our model on the test set ! Because our problem is a regression problem, we have to measure the accuracy of our model with the RMSE (Root Mean Squared Error). Let's see how our model is doing :


```python
final_model = grid_search.best_estimator_

X_test = test_set.drop("charges", axis=1)
y_test = test_set["charges"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse
```




    4592.411834803969



We can also compute a 95% confidence interval for the test RMSE :


```python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```




    array([3615.44349268, 5395.28138953])


