from pprint import pprint
print('Dictionary built in methods')

tweet = {
    'user': 'joelgrus',
    'text': 'Data science is awesome',
    'retweet_count': 100,
    'hashtags': ['#data', '#science', '#datascience', '#awesome', '#yolo']
}

tweet_keys = tweet.keys()
tweet_values = tweet.values()
tweet_items = tweet.items()

pprint(tweet_keys)
pprint(tweet_values)
pprint(tweet_items)

print('#data' in tweet_values)

def assertion_test_arg(some_arg):
    assert some_arg == int, 'you arg is not an int'
    print(some_arg)

assertion_test_arg(1)
assertion_test_arg('bla')
