from pprint import pprint

users = [
    {'id': 0, 'name': 'hero'},
    {'id': 1, 'name': 'Dunn'},
    {'id': 2, 'name': 'Sue'},
    {'id': 3, 'name': 'Chi'},
    {'id': 4, 'name': 'Thor'},
    {'id': 5, 'name': 'Clive'},
    {'id': 6, 'name': 'Hicks'},
    {'id': 7, 'name': 'Devin'},
    {'id': 8, 'name': 'Kate'},
    {'id': 9, 'name': 'Klien'},
]

friendship_pairs = [
    (0,1),
    (0,2),
    (1,2),
    (1,3),
    (2,3),
    (3,4),
    (4,5),
    (5,6),
    (5,7),
    (6,8),
    (7,8),
    (8,9),
]

friendships = {  user['id']: [] for user in users }

for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)


print('Friendship Pairs Dictionary Output')
pprint(friendships)


def number_of_friends(user):
    '''How many friends does a user have?'''
    user_id = user['id']
    friend_ids = friendships[user_id]
    return len(friend_ids)


print('\nHow do we find the total number of connections?')

total_connections = sum(number_of_friends(user) for user in users)
print(total_connections)
