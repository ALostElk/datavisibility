---
configs:
- config_name: yelpreview
  data_files:
  - split: train
    path: yelptrain_data.parquet
  - split: test
    path: yelptest_data.parquet
task_categories:
- summarization
language:
- en
size_categories:
- 100B<n<1T
tags:
- yelp review
- restaurant review
---
# Dataset Card for Yelp Resturant Dataset

## Dataset Description

### Dataset Access
- [Yelp Raw Data Download Link](https://www.yelp.com/dataset/download)

### Raw Dataset Summary
Yelp raw data encompasses a wealth of information from the Yelp platform, detailing user reviews, business ratings, and operational specifics across a diverse array of local establishments. 

To be more specific, yelp raw dataset contains **five different JSON datasets**:
- `yelp_academic_dataset_business.json`  (118.9MB)
This file contains information about businesses listed on Yelp. Each record in this dataset typically includes the business's name, address, city, state, postal code, latitude and longitude, stars (average rating), review count, categories (e.g., Restaurants, Shopping, etc.), and other attributes like parking availability or if it's wheelchair accessible.

- `yelp_academic_dataset_checkin.json` (287MB)
The checkin file provides data on check-ins at businesses by users over time. It includes the business ID and a series of timestamps indicating when users checked in at that location, offering insights into the popularity of the business at different times and days.

- `yelp_academic_dataset_review.json` (5.34GB)
This dataset contains reviews written by users for businesses. Each review includes the user ID, business ID, stars given (1 to 5), useful/funny/cool votes, the text of the review, and the date it was posted. This data can be used to analyze customer sentiment, evaluate service quality, and more.

- `yelp_academic_dataset_tip.json` (180.6MB)
Tips are short messages left by users about a business, often containing suggestions, compliments, or advice for future customers. This file includes information such as the text of the tip, the date it was left, the business ID, and the user ID. Tips provide quick, insightful feedback about a business.

- `yelp_academic_dataset_user.json` (3.36GB)
This file contains data about Yelp users, including their user ID, name, review count, yelping since (the date they joined Yelp), friends (a list of user IDs representing their friends on Yelp), useful/funny/cool vote counts they've received, fans (the number of users who've marked them as a "fan"), and their average stars given. It can be used to analyze user behavior, social networks, and influence on Yelp.

### Language
The Yelp dataset is primarily composed of English language text for its reviews, business information, and user interactions.

## Dataset Process
In this project, we will try only use 
[`yelp_academic_dataset_business.json`](https://yelpdata.s3.us-west-2.amazonaws.com/yelp_academic_dataset_business.json)
 and [`yelp_academic_dataset_review.json`](https://yelpdata.s3.us-west-2.amazonaws.com/yelp_academic_dataset_review.json). (You can check the json files by clicking the links.)

And we will focus solely on restaurants, so we will follow these steps to get our target datasets:
- Load `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` as pandas DataFrames.
- Perform an inner merge of these datasets based on `business_id` and filter out businesses that are not restaurants (filter out rows that `categories` doesn't contain "restaurants").
- Split the yelp restaurants dataset into a training dataset and a testing dataset by shuffling the dataset and then spliting it by 80/20.
- Finally, we get yelp restaurants training dataset and testing dataset.
(Other than doing data processing in .py file, I also provide an individual data processing python file. Please feel free to check if you need: [Data Process Colab Link](https://colab.research.google.com/drive/1r_gUGmsawwtFpZCj23X1jWjfEi6Dw291?usp=sharing))

## Restaurant Dataset

### Restaurant Dataset Summary
- `yelptrain_data.parquet`
This dataset provides a detailed overview of businesses, focusing on aspects such as location, ratings, and customer reviews. It contains columns that identify each business, its geographical information, and metrics indicating its performance, such as aggregate ratings and review counts. Additionally, it includes specifics about the types of services and cuisines offered, operational hours, and detailed customer reviews with ratings, usefulness, humor, and coolness indicators, as well as the text content of the reviews and their posting dates. This dataset includes 3,778,658 rows and it is 2.26 GB.
- `yelptest_data.parquet`
This dataset provides the same information as `yelptrain_data.parquet`, but it includes 943,408 rows and it is 591 MB.

### Supposed Tasks
- Sentiment Analysis: By examining the textual reviews, natural language processing can be used to gauge customer sentiment towards businesses, categorizing opinions into positive, negative, or neutral sentiments.
- Rating Prediction: Machine learning models can leverage user and business attributes to predict the potential ratings a business might receive, helping in understanding factors that influence customer satisfaction.
- Business Analytics: Analysis of business performance metrics such as average ratings, review counts, and operational status can inform business owners about their market standing and customer perceptions.
- Recommendation Systems: The data can feed into recommendation algorithms to suggest businesses to users based on their preferences, previous ratings, and similar user behavior.


### Restaurant Dataset Structure
#### Variables
- business_id: A unique identifier for each business listed in the dataset. (non-null, object)
- name: The name of the business. (non-null, object)
- address: The street address of the business. (non-null, object)
- city: The city where the business is located. (non-null, object)
- state: The state or region where the business is located. (non-null, object)
- postal_code: The postal code associated with the business location. (non-null, object)
- latitude: The latitude coordinate of the business for geographical mapping. (non-null, float64)
- longitude: The longitude coordinate of the business for geographical mapping. (non-null, float64)
- stars_x: The average star rating of the business. (non-null, float64)
- review_count: The number of reviews the business has received. (non-null, int64)
- is_open: A binary variable indicating whether the business is open (1) or closed (0). (non-null, int64)
- attributes: A collection of attributes about the business, like 'Accepts Credit Cards', 'Parking', 'Wi-Fi', etc. (with missing values 493 rows if total 200,000 rows, object)
- categories: The categories the business falls under, such as 'Restaurants', 'Food',’Coffee’, etc. (non-null, object)
- hours: The hours of operation for the business. (with missing values 6,905 rows if total 200,000 rows, object)
- review_id: A unique identifier for each review. (non-null, object)
- user_id: A unique identifier for each user who has left a review. (non-null, object)
- stars_y: The star rating given by the user in their review. (non-null, float64)
- useful: The number of users who found the review useful. (non-null, int64)
- funny: The number of users who found the review funny. (non-null, int64)
- cool: The number of users who found the review cool. (non-null, int64)
- text: The text content of the review. (non-null, object)
- date: The date when the review was posted. (non-null, object)

#### Variables Instances
```
{'business_id': 'XQfwVwDr-v0ZS3_CbbE5Xw',
 'name': 'Turning Point of North Wales',
 'address': '1460 Bethlehem Pike',
 'city': 'North Wales',
 'state': 'PA',
 'postal_code': '19454',
 'latitude': 40.21019744873047,
 'longitude': -75.22364044189453,
 'stars_x': 3.0,
 'review_count': 169.0,
 'is_open': 1.0,
 'categories': 'Restaurants, Breakfast & Brunch, Food, Juice Bars & Smoothies, American (New), Coffee & Tea, Sandwiches',
 'hours': '{"Monday": "7:30-15:0", "Tuesday": "7:30-15:0", "Wednesday": "7:30-15:0", "Thursday": "7:30-15:0", "Friday": "7:30-15:0", "Saturday": "7:30-15:0", "Sunday": "7:30-15:0"}',
 'review_id': 'KU_O5udG6zpxOg-VcAEodg',
 'user_id': 'mh_-eMZ6K5RLWhZyISBhwA',
 'stars_y': 3.0,
 'useful': 0.0,
 'funny': 0.0,
 'cool': 0.0,
 'text': "If you decide to eat here, just be aware it is going to take about 2 hours from beginning to end. We have tried it multiple times, because I want to like it! I have been to it's other locations in NJ and never had a bad experience. \n\nThe food is good, but it takes a very long time to come out. The waitstaff is very young, but usually pleasant. We have just had too many experiences where we spent way too long waiting. We usually opt for another diner or restaurant on the weekends, in order to be done quicker.",
 'date': '2018-07-07 22:09:11',
 'attributes': '{"NoiseLevel": "u\'average\'", "HasTV": "False", "RestaurantsAttire": "\'casual\'", "BikeParking": "False", "Ambience": "{\'touristy\': False, \'hipster\': False, \'romantic\': False, \'divey\': False, \'intimate\': False, \'trendy\': False, \'upscale\': False, \'classy\': False, \'casual\': True}", "WiFi": "\'free\'", "DogsAllowed": "False", "Alcohol": "\'none\'", "BusinessAcceptsCreditCards": "True", "RestaurantsGoodForGroups": "True", "RestaurantsPriceRange2": "2", "RestaurantsReservations": "False", "WheelchairAccessible": "True", "BusinessAcceptsBitcoin": "False", "RestaurantsTableService": "True", "GoodForKids": "True", "Caters": "False", "HappyHour": "False", "RestaurantsDelivery": "True", "GoodForMeal": "{\'dessert\': False, \'latenight\': False, \'lunch\': True, \'dinner\': False, \'brunch\': True, \'breakfast\': True}", "OutdoorSeating": "True", "RestaurantsTakeOut": "True", "BusinessParking": "{\'garage\': False, \'street\': False, \'validated\': False, \'lot\': True, \'valet\': False}"}'}
```
### Usage
The dataset is compatible with the Hugging Face `datasets` library. The dataset class `YelpDataset` provides methods to access the structured data efficiently, including features detailing business information, user reviews, and user profiles.

### Getting Started
To start working with the Yelp Dataset in Python, ensure you have the Hugging Face `datasets` library installed. Then, you can load the dataset using the `YelpDataset` class provided in the script. Here's a quick example:
```
from datasets import load_dataset

dataset = load_dataset("Johnnyeee/Yelpdata_663", trust_remote_code=True)
```
This will give you a quick glimpse into the structure and content of the dataset, ready for your analysis or model training tasks.

You can also generate a training dataset example by:
```
next(iter((dataset['train'])))
```
A testing dataset example
```
next(iter((dataset['test'])))
```

You can check this Colab link to find out more details: [Link*](https://colab.research.google.com/drive/1ybXGIYUqJ7DH22A4apynfrWCMGzb2v_T?usp=sharing)

## Dataset Creation
### Curation Rationale
The dataset includes a variety of data types (e.g., business information, reviews, user data, check-ins, and tips), enabling a wide range of research topics and studies in areas such as natural language processing, social network analysis, recommender systems, and geographic information systems.

By providing data from an active and popular platform, the dataset offers insights into real-world consumer behavior, business trends, and social interactions. This relevance makes it an excellent resource for studies aiming to understand or model aspects of the contemporary economy and society.

By making the dataset publicly available for academic and educational purposes, Yelp aims to contribute to the broader academic community. It lowers barriers for researchers and educators who might not have access to large-scale, real-world data.

## Considerations
### Bias
- Geographic Bias: Yelp's presence and popularity vary significantly across different regions. If the dataset has more extensive coverage in certain areas, the analysis might not accurately reflect regions with lower Yelp usage, leading to skewed insights about restaurant preferences or trends.
- User Demographic Bias: Yelp users may not be a representative sample of the broader population. Factors such as age, income, and tech-savviness can influence who uses Yelp and who writes reviews. This skew can affect the perceived quality or popularity of restaurants.
- Selection Bias: By focusing solely on restaurants and the first 200,000 rows of the merged dataset, there's a risk of omitting relevant data that could offer a more comprehensive understanding of consumer preferences or business performance. The initial selection process might also favor certain types of restaurants or those with more reviews, skewing the analysis.
- Rating Bias: Users who leave reviews might be more likely to do so after exceptionally positive or negative experiences, which doesn't always accurately reflect the average customer experience. This can lead to a polarization of ratings, where the data might not accurately represent the overall quality of service.

### Limitations
- Data Completeness: The dataset might not capture all restaurants or reviews, especially newer businesses or those that have not been reviewed on Yelp. This incompleteness can limit the analysis's scope and the accuracy of findings.
- Temporal Dynamics: Consumer preferences and restaurant quality can change over time. The dataset represents a snapshot, and without considering the time aspect, it might not accurately reflect current trends or the impact of external events (e.g., a pandemic).
- Memory Constraints: Limiting the analysis to the first 200,000 rows to manage memory usage could introduce sample bias, as this approach does not guarantee a random or representative sample of the entire dataset. This constraint might overlook valuable insights from the excluded data.
- Lack of External Data: By not incorporating external data sources, such as economic indicators, health inspection scores, or social media sentiment, the analysis might miss out on important factors that could influence restaurant performance or consumer preferences.
- Data Privacy and Ethics: While the dataset is curated for academic use, there's always a concern regarding user privacy and the ethical use of data, particularly in how user-generated content is analyzed and interpreted.

### Dataset Terms of Use
Yelp's dataset comes with a detailed set of terms of use, which you can review by visiting their Dataset User Agreement. The agreement can be found at the provided link: [Yelp Dataset User Agreement](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf). This document will contain specific guidelines and restrictions that are crucial for anyone working with Yelp's dataset.


# Links
All relative links:
- Yelp raw dataset: https://www.yelp.com/dataset/download
- yelp_academic_dataset_business.json: https://yelpdata.s3.us-west-2.amazonaws.com/yelp_academic_dataset_business.json
- yelp_academic_dataset_review.json: https://yelpdata.s3.us-west-2.amazonaws.com/yelp_academic_dataset_review.json
- Data Processing: https://colab.research.google.com/drive/1r_gUGmsawwtFpZCj23X1jWjfEi6Dw291?usp=sharing
- Dataset Check: https://colab.research.google.com/drive/1ybXGIYUqJ7DH22A4apynfrWCMGzb2v_T?usp=sharing