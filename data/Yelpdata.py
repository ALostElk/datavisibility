# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os
from typing import List
import datasets
import logging
from random import random

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://www.yelp.com/dataset/download"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""


class YelpDataset(datasets.GeneratorBasedBuilder):
    """Yelp Dataset focusing on restaurant reviews and business information."""
    
    VERSION = datasets.Version("1.1.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="restaurants", version=VERSION, description="This part of the dataset covers a wide range of restaurants"),
    ]

    DEFAULT_CONFIG_NAME = "restaurants"
    
    _URL = "https://yelpdata.s3.us-west-2.amazonaws.com/"
    _URLS = {
        "business": _URL + "yelp_academic_dataset_business.json",
        "review": _URL + "yelp_academic_dataset_review.json",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "business_id": datasets.Value("string"),
                "name": datasets.Value("string"),
                "address": datasets.Value("string"),
                "city": datasets.Value("string"),
                "state": datasets.Value("string"),
                "postal_code": datasets.Value("string"),
                "latitude": datasets.Value("float"),
                "longitude": datasets.Value("float"),
                "stars_x": datasets.Value("float"),
                "review_count": datasets.Value("float"),
                "is_open": datasets.Value("float"),
                "categories": datasets.Value("string"),
                "hours": datasets.Value("string"),
                "review_id": datasets.Value("string"),
                "user_id": datasets.Value("string"),
                "stars_y": datasets.Value("float"),
                "useful": datasets.Value("float"),
                "funny": datasets.Value("float"),
                "cool": datasets.Value("float"),
                "text": datasets.Value("string"),
                "date": datasets.Value("string"),
                "attributes": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage="https://www.yelp.com/dataset/download",
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(self._URLS)
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": downloaded_files, "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"files": downloaded_files, "split": "test"}),
        ]


    def _generate_examples(self, files, split):
        """Yields examples as (key, example) tuples."""
        business_path, review_path = files["business"], files["review"]
    
        # Load businesses and filter for restaurants
        with open(business_path, encoding="utf-8") as f:
            businesses = {}
            for line in f:
                business = json.loads(line)
            # Check if 'categories' is not None and contains "Restaurants"
                if business.get("categories") and "Restaurants" in business["categories"]:
                    businesses[business['business_id']] = business
    
    # Generate examples with an attempted 80/20 split for train/test
        with open(review_path, encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)
                business_id = review['business_id']
                if business_id in businesses:
                    business = businesses[business_id]
                    example = {
                        "business_id": business['business_id'],
                        "name": business.get("name", ""),
                        "address": business.get("address", ""),
                        "city": business.get("city", ""),
                        "state": business.get("state", ""),
                        "postal_code": business.get("postal_code", ""),
                        "latitude": business.get("latitude", None),
                        "longitude": business.get("longitude", None),
                        "stars_x": business.get("stars", None),
                        "review_count": business.get("review_count", None),
                        "is_open": business.get("is_open", None),
                        "categories": business.get("categories", ""),
                        "hours": json.dumps(business.get("hours", {})),  # Storing hours as a JSON string
                        "review_id": review.get("review_id", ""),
                        "user_id": review.get("user_id", ""),
                        "stars_y": review.get("stars", None),
                        "useful": review.get("useful", None),
                        "funny": review.get("funny", None),
                        "cool": review.get("cool", None),
                        "text": review.get("text", ""),
                        "date": review.get("date", ""),
                        "attributes": json.dumps(business.get("attributes", {})),  # Storing attributes as a JSON string
                    }
                # Randomly assign to split based on an 80/20 ratio
                    if (split == 'train' and random() < 0.8) or (split == 'test' and random() >= 0.8):
                        yield review['review_id'], example
