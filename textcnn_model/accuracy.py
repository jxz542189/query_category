import pandas as pd


category_id = "601"
for category_id in ["601", "602", "603", "604", "605", "606", "607", "608", "609", "610",
                    "611", "612", "613", "614", "615", "616", "617", "618", "619", "620",
                    "621", "622", "623", "624", "625", "626", "627", "628", "629", "635",
                    "637", "646", "647", "651"]:
    df = pd.read_csv(f"test/{category_id}.csv")
    total = df.shape[0]
    equal_total = df.query('label == predict').shape[0]
    unequal_df = df.query('label != predict')
    unequal_df.to_csv(f"test/{category_id}_unequal.csv", index=None)
    print(f"{category_id} accuracy: {equal_total / total}")