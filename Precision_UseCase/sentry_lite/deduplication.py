

# sentry_lite/deduplication.py 
from fuzzywuzzy import fuzz 


def match_score(name1, name2): 

    return fuzz.token_sort_ratio(name1, name2) 

 

def deduplicate(sponsor_df, new_record): 

    results = [] 

    for _, row in sponsor_df.iterrows(): 

        score = match_score(row["Sponsor_ID"], new_record["Sponsor_ID"]) 

        if score > 85: 

            results.append((row["Sponsor_ID"], score)) 

    return results 

 
