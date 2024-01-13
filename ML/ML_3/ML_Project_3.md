# Calculating EA Sports FC 24 Player Ratings based on their Stats :

Ever since I was a kid, I have always played the FIFA games on my computer or PS4. In the FIFA games, each player has an "Overall Rating": it is a number that represents how "good" he is in the game. Each player also has many other ratings that give us an idea of the strengths of a certain player.
But how are those Overall "Ratings" calculated ? That was always an intereting question for me.  
In this project, the aim is to see if we can predict accurately player's overall ratings. We will use different methods :

- **First method**: the "Overall Rating" will be calculated as a linear combination of all his ratings
- **Second method**: we will try to predict a player's "Overall Rating" with IRL statistics of the players for the previous season (not sure if this is possible, but we will give it a go !)

# Part 1: Linear Regression with the sub-ratings
  
In the first part of this project, we will first load our dataset and then work with our data using a linear classifier.

## Loading and preprocessing the data:

First we load the data, which is a dataset that I have already downloaded (from Kaggle).


```python
import pandas as pd

file_path="./datasets/archive/male_players.csv"

ea_stats = pd.read_csv(file_path)

ea_stats.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
      max-width: 800px; /* Adjust the max-width as needed */
        overflow-x: scroll;
    }

    .dataframe tbody tr th {
        vertical-align: top;
        max-width: 800px; /* Adjust the max-width as needed */
        overflow-x: scroll;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Nation</th>
      <th>Club</th>
      <th>Position</th>
      <th>Age</th>
      <th>Overall</th>
      <th>Pace</th>
      <th>Shooting</th>
      <th>Passing</th>
      <th>...</th>
      <th>Strength</th>
      <th>Aggression</th>
      <th>Att work rate</th>
      <th>Def work rate</th>
      <th>Preferred foot</th>
      <th>Weak foot</th>
      <th>Skill moves</th>
      <th>URL</th>
      <th>Gender</th>
      <th>GK</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Kylian Mbappé</td>
      <td>France</td>
      <td>Paris SG</td>
      <td>ST</td>
      <td>24</td>
      <td>91</td>
      <td>97</td>
      <td>90</td>
      <td>80</td>
      <td>...</td>
      <td>77</td>
      <td>64</td>
      <td>High</td>
      <td>Low</td>
      <td>Right</td>
      <td>4</td>
      <td>5</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Erling Haaland</td>
      <td>Norway</td>
      <td>Manchester City</td>
      <td>ST</td>
      <td>23</td>
      <td>91</td>
      <td>89</td>
      <td>93</td>
      <td>66</td>
      <td>...</td>
      <td>93</td>
      <td>87</td>
      <td>High</td>
      <td>Medium</td>
      <td>Left</td>
      <td>3</td>
      <td>3</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Kevin De Bruyne</td>
      <td>Belgium</td>
      <td>Manchester City</td>
      <td>CM</td>
      <td>32</td>
      <td>91</td>
      <td>72</td>
      <td>88</td>
      <td>94</td>
      <td>...</td>
      <td>74</td>
      <td>75</td>
      <td>High</td>
      <td>Medium</td>
      <td>Right</td>
      <td>5</td>
      <td>4</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Lionel Messi</td>
      <td>Argentina</td>
      <td>Inter Miami CF</td>
      <td>CF</td>
      <td>36</td>
      <td>90</td>
      <td>80</td>
      <td>87</td>
      <td>90</td>
      <td>...</td>
      <td>68</td>
      <td>44</td>
      <td>Low</td>
      <td>Low</td>
      <td>Left</td>
      <td>4</td>
      <td>4</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Karim Benzema</td>
      <td>France</td>
      <td>Al Ittihad</td>
      <td>CF</td>
      <td>35</td>
      <td>90</td>
      <td>79</td>
      <td>88</td>
      <td>83</td>
      <td>...</td>
      <td>82</td>
      <td>63</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>Right</td>
      <td>4</td>
      <td>4</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Thibaut Courtois</td>
      <td>Belgium</td>
      <td>Real Madrid</td>
      <td>GK</td>
      <td>31</td>
      <td>90</td>
      <td>85</td>
      <td>89</td>
      <td>76</td>
      <td>...</td>
      <td>70</td>
      <td>23</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>Left</td>
      <td>3</td>
      <td>1</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Harry Kane</td>
      <td>England</td>
      <td>FC Bayern München</td>
      <td>ST</td>
      <td>30</td>
      <td>90</td>
      <td>69</td>
      <td>93</td>
      <td>84</td>
      <td>...</td>
      <td>84</td>
      <td>80</td>
      <td>High</td>
      <td>High</td>
      <td>Right</td>
      <td>5</td>
      <td>3</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Robert Lewandowski</td>
      <td>Poland</td>
      <td>FC Barcelona</td>
      <td>ST</td>
      <td>35</td>
      <td>90</td>
      <td>75</td>
      <td>91</td>
      <td>80</td>
      <td>...</td>
      <td>89</td>
      <td>81</td>
      <td>High</td>
      <td>Medium</td>
      <td>Right</td>
      <td>4</td>
      <td>4</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Mohamed Salah</td>
      <td>Egypt</td>
      <td>Liverpool</td>
      <td>RW</td>
      <td>31</td>
      <td>89</td>
      <td>89</td>
      <td>87</td>
      <td>81</td>
      <td>...</td>
      <td>75</td>
      <td>63</td>
      <td>High</td>
      <td>Medium</td>
      <td>Left</td>
      <td>3</td>
      <td>4</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>Rúben Dias</td>
      <td>Portugal</td>
      <td>Manchester City</td>
      <td>CB</td>
      <td>26</td>
      <td>89</td>
      <td>62</td>
      <td>39</td>
      <td>66</td>
      <td>...</td>
      <td>90</td>
      <td>93</td>
      <td>Medium</td>
      <td>High</td>
      <td>Right</td>
      <td>4</td>
      <td>2</td>
      <td>https://www.ea.com/games/ea-sports-fc/ratings/...</td>
      <td>M</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 47 columns</p>
</div>



Let's take a look at our data: we can plot some histograms to get an idea of the distribution for each rating.


```python
%matplotlib inline
import matplotlib.pyplot as plt
ea_stats.hist(bins=50, figsize=(20,15))
plt.show()
```


    
![png](ML/ML_3/images/output_6_0.png)
    


That looks very messy... Let's clean our dataset to have a clearer place to work!  
We will drop certain columns (or features) that don't give us useful information for the "Overall Rating" of a player, like the "Preferred Foot" or "Gender" (because all players all male)... We also want to get rid of all goalkeepers, because it is a special position to play and less statistics will be available for the next part of the project :)


```python
dropped_columns = ["Name","Nation","Club","Preferred foot","URL","GK","Gender","Att work rate","Def work rate","Unnamed: 0"]

X = ea_stats.drop(dropped_columns,axis=1)
X = X[ea_stats["Position"] != "GK"].reset_index(drop=True)
y = X["Overall"]
X.dropna(axis=0)
kept_columns = X.columns
X.head(10)
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
      <th>Position</th>
      <th>Age</th>
      <th>Overall</th>
      <th>Pace</th>
      <th>Shooting</th>
      <th>Passing</th>
      <th>Dribbling</th>
      <th>Defending</th>
      <th>Physicality</th>
      <th>Acceleration</th>
      <th>...</th>
      <th>Heading</th>
      <th>Def</th>
      <th>Standing</th>
      <th>Sliding</th>
      <th>Jumping</th>
      <th>Stamina</th>
      <th>Strength</th>
      <th>Aggression</th>
      <th>Weak foot</th>
      <th>Skill moves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ST</td>
      <td>24</td>
      <td>91</td>
      <td>97</td>
      <td>90</td>
      <td>80</td>
      <td>93</td>
      <td>36</td>
      <td>78</td>
      <td>97</td>
      <td>...</td>
      <td>73</td>
      <td>26</td>
      <td>34</td>
      <td>32</td>
      <td>88</td>
      <td>88</td>
      <td>77</td>
      <td>64</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ST</td>
      <td>23</td>
      <td>91</td>
      <td>89</td>
      <td>93</td>
      <td>66</td>
      <td>79</td>
      <td>45</td>
      <td>88</td>
      <td>82</td>
      <td>...</td>
      <td>83</td>
      <td>38</td>
      <td>47</td>
      <td>29</td>
      <td>93</td>
      <td>76</td>
      <td>93</td>
      <td>87</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CM</td>
      <td>32</td>
      <td>91</td>
      <td>72</td>
      <td>88</td>
      <td>94</td>
      <td>86</td>
      <td>65</td>
      <td>78</td>
      <td>72</td>
      <td>...</td>
      <td>55</td>
      <td>66</td>
      <td>70</td>
      <td>53</td>
      <td>72</td>
      <td>88</td>
      <td>74</td>
      <td>75</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CF</td>
      <td>36</td>
      <td>90</td>
      <td>80</td>
      <td>87</td>
      <td>90</td>
      <td>96</td>
      <td>33</td>
      <td>64</td>
      <td>87</td>
      <td>...</td>
      <td>60</td>
      <td>20</td>
      <td>35</td>
      <td>24</td>
      <td>71</td>
      <td>70</td>
      <td>68</td>
      <td>44</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CF</td>
      <td>35</td>
      <td>90</td>
      <td>79</td>
      <td>88</td>
      <td>83</td>
      <td>87</td>
      <td>39</td>
      <td>78</td>
      <td>78</td>
      <td>...</td>
      <td>90</td>
      <td>43</td>
      <td>24</td>
      <td>18</td>
      <td>85</td>
      <td>82</td>
      <td>82</td>
      <td>63</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ST</td>
      <td>30</td>
      <td>90</td>
      <td>69</td>
      <td>93</td>
      <td>84</td>
      <td>82</td>
      <td>49</td>
      <td>83</td>
      <td>67</td>
      <td>...</td>
      <td>89</td>
      <td>46</td>
      <td>46</td>
      <td>38</td>
      <td>87</td>
      <td>83</td>
      <td>84</td>
      <td>80</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ST</td>
      <td>35</td>
      <td>90</td>
      <td>75</td>
      <td>91</td>
      <td>80</td>
      <td>86</td>
      <td>44</td>
      <td>84</td>
      <td>76</td>
      <td>...</td>
      <td>91</td>
      <td>35</td>
      <td>42</td>
      <td>19</td>
      <td>92</td>
      <td>76</td>
      <td>89</td>
      <td>81</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RW</td>
      <td>31</td>
      <td>89</td>
      <td>89</td>
      <td>87</td>
      <td>81</td>
      <td>88</td>
      <td>45</td>
      <td>76</td>
      <td>89</td>
      <td>...</td>
      <td>59</td>
      <td>38</td>
      <td>43</td>
      <td>41</td>
      <td>80</td>
      <td>87</td>
      <td>75</td>
      <td>63</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CB</td>
      <td>26</td>
      <td>89</td>
      <td>62</td>
      <td>39</td>
      <td>66</td>
      <td>64</td>
      <td>89</td>
      <td>87</td>
      <td>54</td>
      <td>...</td>
      <td>87</td>
      <td>91</td>
      <td>91</td>
      <td>87</td>
      <td>84</td>
      <td>78</td>
      <td>90</td>
      <td>93</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LW</td>
      <td>23</td>
      <td>89</td>
      <td>95</td>
      <td>82</td>
      <td>78</td>
      <td>92</td>
      <td>29</td>
      <td>68</td>
      <td>95</td>
      <td>...</td>
      <td>50</td>
      <td>32</td>
      <td>25</td>
      <td>18</td>
      <td>74</td>
      <td>84</td>
      <td>64</td>
      <td>58</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 37 columns</p>
</div>



Let's also take a look at the "Overall Rating" vector y, to check if we have what we wanted:


```python
y.head(10)
```




    0    91
    1    91
    2    91
    3    90
    4    90
    5    90
    6    90
    7    89
    8    89
    9    89
    Name: Overall, dtype: int64



# Create our Linear Model and make Predictions:

As we already said in the beginning, we want to compute the "Overall Rating" with the other sub-ratings. In theory, this model should be doing a good job (how else are they supposed to get the "Overall Rating" ??). Let's first build a small and very basic pipeline to "code" the position feature to a number, as models usually don't like non-numerical values.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

col_to_enc = ["Position"]

pipeline = Pipeline(steps=[
    ("encoder", OrdinalEncoder(dtype="int64"))
])

col_encd = pipeline.fit_transform(X[["Position"]])
X.Position = col_encd
X.head()
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
      <th>Position</th>
      <th>Age</th>
      <th>Overall</th>
      <th>Pace</th>
      <th>Shooting</th>
      <th>Passing</th>
      <th>Dribbling</th>
      <th>Defending</th>
      <th>Physicality</th>
      <th>Acceleration</th>
      <th>...</th>
      <th>Heading</th>
      <th>Def</th>
      <th>Standing</th>
      <th>Sliding</th>
      <th>Jumping</th>
      <th>Stamina</th>
      <th>Strength</th>
      <th>Aggression</th>
      <th>Weak foot</th>
      <th>Skill moves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>24</td>
      <td>91</td>
      <td>97</td>
      <td>90</td>
      <td>80</td>
      <td>93</td>
      <td>36</td>
      <td>78</td>
      <td>97</td>
      <td>...</td>
      <td>73</td>
      <td>26</td>
      <td>34</td>
      <td>32</td>
      <td>88</td>
      <td>88</td>
      <td>77</td>
      <td>64</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>23</td>
      <td>91</td>
      <td>89</td>
      <td>93</td>
      <td>66</td>
      <td>79</td>
      <td>45</td>
      <td>88</td>
      <td>82</td>
      <td>...</td>
      <td>83</td>
      <td>38</td>
      <td>47</td>
      <td>29</td>
      <td>93</td>
      <td>76</td>
      <td>93</td>
      <td>87</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>32</td>
      <td>91</td>
      <td>72</td>
      <td>88</td>
      <td>94</td>
      <td>86</td>
      <td>65</td>
      <td>78</td>
      <td>72</td>
      <td>...</td>
      <td>55</td>
      <td>66</td>
      <td>70</td>
      <td>53</td>
      <td>72</td>
      <td>88</td>
      <td>74</td>
      <td>75</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>36</td>
      <td>90</td>
      <td>80</td>
      <td>87</td>
      <td>90</td>
      <td>96</td>
      <td>33</td>
      <td>64</td>
      <td>87</td>
      <td>...</td>
      <td>60</td>
      <td>20</td>
      <td>35</td>
      <td>24</td>
      <td>71</td>
      <td>70</td>
      <td>68</td>
      <td>44</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>35</td>
      <td>90</td>
      <td>79</td>
      <td>88</td>
      <td>83</td>
      <td>87</td>
      <td>39</td>
      <td>78</td>
      <td>78</td>
      <td>...</td>
      <td>90</td>
      <td>43</td>
      <td>24</td>
      <td>18</td>
      <td>85</td>
      <td>82</td>
      <td>82</td>
      <td>63</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



Now that's better ! Let's now create sets for training and for testing :


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

X_train.head()
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
      <th>Position</th>
      <th>Age</th>
      <th>Overall</th>
      <th>Pace</th>
      <th>Shooting</th>
      <th>Passing</th>
      <th>Dribbling</th>
      <th>Defending</th>
      <th>Physicality</th>
      <th>Acceleration</th>
      <th>...</th>
      <th>Heading</th>
      <th>Def</th>
      <th>Standing</th>
      <th>Sliding</th>
      <th>Jumping</th>
      <th>Stamina</th>
      <th>Strength</th>
      <th>Aggression</th>
      <th>Weak foot</th>
      <th>Skill moves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10979</th>
      <td>13</td>
      <td>31</td>
      <td>62</td>
      <td>80</td>
      <td>62</td>
      <td>45</td>
      <td>59</td>
      <td>21</td>
      <td>61</td>
      <td>77</td>
      <td>...</td>
      <td>60</td>
      <td>25</td>
      <td>12</td>
      <td>17</td>
      <td>76</td>
      <td>64</td>
      <td>69</td>
      <td>35</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9518</th>
      <td>2</td>
      <td>29</td>
      <td>64</td>
      <td>68</td>
      <td>41</td>
      <td>56</td>
      <td>60</td>
      <td>60</td>
      <td>68</td>
      <td>67</td>
      <td>...</td>
      <td>50</td>
      <td>61</td>
      <td>61</td>
      <td>59</td>
      <td>64</td>
      <td>71</td>
      <td>69</td>
      <td>63</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2916</th>
      <td>13</td>
      <td>34</td>
      <td>72</td>
      <td>65</td>
      <td>72</td>
      <td>68</td>
      <td>69</td>
      <td>44</td>
      <td>71</td>
      <td>63</td>
      <td>...</td>
      <td>69</td>
      <td>45</td>
      <td>38</td>
      <td>26</td>
      <td>76</td>
      <td>68</td>
      <td>76</td>
      <td>61</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9161</th>
      <td>1</td>
      <td>29</td>
      <td>64</td>
      <td>53</td>
      <td>32</td>
      <td>46</td>
      <td>39</td>
      <td>63</td>
      <td>73</td>
      <td>52</td>
      <td>...</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>61</td>
      <td>74</td>
      <td>67</td>
      <td>82</td>
      <td>56</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9314</th>
      <td>0</td>
      <td>30</td>
      <td>64</td>
      <td>62</td>
      <td>62</td>
      <td>65</td>
      <td>64</td>
      <td>47</td>
      <td>65</td>
      <td>64</td>
      <td>...</td>
      <td>44</td>
      <td>39</td>
      <td>52</td>
      <td>48</td>
      <td>55</td>
      <td>71</td>
      <td>64</td>
      <td>60</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



Now it's time to create our model ! We will use the **LinearRegression** model from sklearn, one of the most simple model.


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def linear_regression_mae(X_train,y_train,X_test,y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(predictions, y_test)
    return(mae,predictions)
```

Now we can make some predictions :


```python
mae ,predictions =linear_regression_mae(X_train, y_train, X_test, y_test)

print(mae)
```

    3.537805420831139e-14


The Mean Absolute Error (MAE) is very small, which means our models is doing some good predictions. And because the MAE is so low, we can thnik that the "Overall Rating" is indeed linearly computed from the other ratings.

We can also compare the real ratings with the predicted ratings to show how close we are :


```python
y_test_head = list(y_test)[:20]

print(y_test_head)
```

    [61, 68, 66, 86, 69, 58, 72, 59, 78, 65, 69, 55, 75, 50, 59, 75, 59, 51, 68, 54]



```python
predictions = [int(item) for item in predictions]
pred_list_head = predictions[:20]

print(pred_list_head)
```

    [60, 67, 65, 85, 68, 57, 71, 58, 77, 64, 68, 54, 75, 49, 58, 74, 58, 50, 67, 53]



```python
abs = range(1,21)

plt.plot(y_test_head)
plt.plot(pred_list_head)
plt.show()
```


    
![png](ML/ML_3/images/output_24_0.png)
    


We can see that for high ratings (above 80) and low ratings (below 60) the model is giving quite accurate results, usually one point off maximum.

Now we can try something else, which is already a bit more fun. A FIFA player card usually looks like this:

![Haaland-2.png](ML/ML_3/images/output_37_0.png)

We can see the "Overall Rating" of the player, which is 96; but we also see 6 other ratings a the bottom of the card (Pace, Shooting, Passing, Dribbling, Defending and Physicality). Those ratings are already calculated from other sub-ratings we have work with previously, so calculating those 6 ratings separatly should be quite easy.    

  
Now my question is : can we have an accurate prediction of the "Overall Rating" using only those 6 ratings as training data ? Let's find out !

First we get rid of all the other columns :


```python
kept_columns_2 = ["Pace","Shooting","Passing","Dribbling","Defending","Physicality"]

X_2 = X[kept_columns_2]

X_2.head()
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
      <th>Pace</th>
      <th>Shooting</th>
      <th>Passing</th>
      <th>Dribbling</th>
      <th>Defending</th>
      <th>Physicality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>97</td>
      <td>90</td>
      <td>80</td>
      <td>93</td>
      <td>36</td>
      <td>78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>89</td>
      <td>93</td>
      <td>66</td>
      <td>79</td>
      <td>45</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>88</td>
      <td>94</td>
      <td>86</td>
      <td>65</td>
      <td>78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>87</td>
      <td>90</td>
      <td>96</td>
      <td>33</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79</td>
      <td>88</td>
      <td>83</td>
      <td>87</td>
      <td>39</td>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>



Like previously, we create the sets for training and testing: 


```python
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2,random_state=0)
```


```python
mae_2,predictions_2 =linear_regression_mae(X_train, y_train, X_test, y_test)

print(mae_2)
```

    2.8887184357258002


**Comments**: the MAE is higher, which shows us it is harder to make a prediction with only 6 ratings than with all the ratings.

Again, we can visualise the predictions and the real "Overall Ratings" on a graph :


```python
print(list(y_test)[:20])
predictions_2 = [int(item) for item in predictions_2]
print(predictions_2[:20])
```

    [61, 68, 66, 86, 69, 58, 72, 59, 78, 65, 69, 55, 75, 50, 59, 75, 59, 51, 68, 54]
    [60, 67, 65, 74, 66, 67, 74, 62, 73, 63, 67, 55, 71, 57, 61, 75, 61, 53, 66, 59]



```python
plt.plot(list(y_test)[:20])
plt.plot(predictions_2[:20])
plt.show()
```


    
![png](output_37_0.png)
    


Less precise than the previous the previous case...

# Part 2: Using the IRL Statistics of the players

In this second part we will try (maybe with success !) to predict the "Overall Rating" of players with their statistics from the previous season.

(to be continued...)
