#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:27:43 2017

@author: yvan
"""

def escape(a_string):
    if a_string is not None:
        return a_string.replace("'", "").replace("\\", "")
    return ''

import twitter
from neo4j.v1 import GraphDatabase
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
    
CONSUMER_KEY = cfg['twitter']['CONSUMER_KEY']
CONSUMER_SECRET = cfg['twitter']['CONSUMER_SECRET']
ACCESS_TOKEN = cfg['twitter']['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = cfg['twitter']['ACCESS_TOKEN_SECRET']

api = twitter.Api(CONSUMER_KEY,
          CONSUMER_SECRET,
          ACCESS_TOKEN,
          ACCESS_TOKEN_SECRET)

uri = "bolt://localhost:7687"
neo4j = GraphDatabase.driver(uri, auth=("neo4j", "root"))
session = neo4j.session()

tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

# http://boundingbox.klokantech.com/
# A list of Longitude,Latitude pairs specifying bounding boxes for the tweets’ origin
FRANCE = ['-5.0006170181','42.1758579539', '8.6452145941','51.1825646283']
print("Waiting 4 tweet")
nbTweets = 0
for tweet in api.GetStreamFilter(locations=FRANCE):
    #print("NEW TWEET =============================================")
    #print(tweet)
    try:
        lang = tweet['lang']
        if lang != 'fr':
            continue
        tweet_id = str(tweet['id'])
        user_id  = str(tweet['user']['id']) 
        text = tweet['text']
        blob = tb(text)
        polarity, subjectivity = blob.sentiment
        print("Inserting new messsage {} from {}".format(tweet_id, user_id))
        request = """
        CREATE (t:TWEET {{ ID:{tweet_id}, TEXT:'{text}', POLARITY:{polarity}, SUBJECTIVITY:{subjectivity} }})
        MERGE (u:USER {{ID:{user_id}}}) SET u.NAME='{user_name}', u.PSEUDO='{pseudo}', u.LOCATION='{location}'
        CREATE (u)-[:WRITE]->(t)
        """.format(tweet_id = tweet_id, text = escape(text), polarity = polarity, subjectivity = subjectivity,
        user_id = user_id, user_name = escape(tweet['user']['name']), pseudo = escape(tweet['user']['name']), location = escape(tweet['user']['location']))
        
        if tweet['in_reply_to_status_id']:
            original_tweet = str(tweet['in_reply_to_status_id'])
            request += """
        MERGE (o:TWEET {{ID:{reply_to}}})
        CREATE (t)-[:REPLY]->(o)
                """.format(reply_to=original_tweet)
            if tweet['in_reply_to_status_id']:
                original_user = str(tweet['in_reply_to_user_id'])
                request += """
        MERGE (x:USER {{ID:{reply_to}}})
        MERGE (x)-[:WRITE]->(o)
                """.format(reply_to=original_user)
                print(user_id + " reply to " + str(original_user) + " about " + str(original_tweet))
                
        
        if tweet['entities'] and tweet['entities']['user_mentions'] and len(tweet['entities']['user_mentions']) > 0:
            for mention in tweet['entities']['user_mentions']:
                request += """
        MERGE (m{mention_to}:USER {{ID:{mention_to}}})
        CREATE (t)-[:MENTION]->(m{mention_to})
        """.format(mention_to=mention['id'])
        
        # print("Executing = " + request)
        result = session.run(request)
        # print(result)
        nbTweets += 1
        if nbTweets % 10 == 0:
            print("Number of tweets : " + str(nbTweets))    
    except Exception as e:
        print("Exception: {0}".format(e))
        print(tweet)
        
