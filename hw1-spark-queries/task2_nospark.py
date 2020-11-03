#GOAL: Compute average stars for each business category and output top n categories with the highest
# average stars. Do not use PySpark

import json 

def inv_index(review_file):
    '''Creates inverted index on review.json using business_id as key
    review_file: an input .json file with one json object per review
    inv_index key: business_id
    inv_index vals: list of {review_id : stars}
    '''
    review_list = []
    inv_index = {}
    with open(review_file) as json_file:
        for json_obj in json_file:
            review_dict = json.loads(json_obj)
            b_id = review_dict['business_id']
            if b_id not in inv_index:
                inv_index[b_id] = [(review_dict['review_id'], review_dict['stars'])]
            else:
                inv_index[b_id].append((review_dict['review_id'], review_dict['stars']))
    return inv_index

def category_avg(business_file, review_inv_index, n):
    '''For each business in business.json, get the average stars. Then for each
    business"s category, add and update the average. Keep re-calculating each category star average
    for each busienss. 
    business_file: business.json containing one json object per business
    review_inv_index: inv_index created in first function
    n: the top n categories with the highest star'''
    category = {}
    with open(business_file) as json_file:
        for json_obj in json_file:
            business_obj = json.loads(json_obj)
            b_id = business_obj['business_id']

            #Get categories
            categories = business_obj['categories']
            if categories == None:
                continue
            else:
                cat_list = business_obj['categories'].split(', ')

            #Get business review count and sum
            if b_id in review_inv_index: #Reviews found
                review_stars = review_inv_index[b_id]
                review_count = len(review_stars) #Get review_count

                stars = [review[1] for review in review_stars] #Get review_sum
                review_sum = sum(stars)

                review_avg = review_sum / review_count #Get review_avg

            else: #No reviews found
                review_avg = None

            #Accumulate each business count and sum by category
            for cat in cat_list:
                if cat not in category: #net-new category
                    new_avg = review_avg
                    category[cat] = review_avg
                    
                else: #update category
                    prev_avg = category[cat]
                    if prev_avg == None and review_avg == None:
                        category[cat] = review_avg
                    elif prev_avg == None and review_avg != None:
                        category[cat] = review_avg
                    elif prev_avg != None and review_avg == None:
                        pass
                    elif prev_avg != None and review_avg != None:
                        new_avg = (prev_avg + review_avg) / 2
                        category[cat] = new_avg
                        
            #Select top n categories
            cat_tups = list(category.items())
            cat_valid = [tup for tup in cat_tups if tup[1] != None]
            cat_valid.sort(key=lambda x: (-x[1], x[0]))
            cat_valid = cat_valid[:n]
            
            #Formatting requirement
            result = [list(item) for item in cat_valid]
            result = {"result": result}
    return result

def main():
    review_file = 'data/review.json'
    review_inv_index = inv_index(review_file)
    business_file = 'data/business.json'
    n = 20
    result = category_avg(business_file, review_inv_index, n)
    print(result)

if __name__ == '__main__':
    main()