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
import sys
import twitter
from neo4j.v1 import GraphDatabase
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import os

sys.stderr.write("Reading properties\n")

CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']

sys.stderr.write("Waiting for TWITTER\n")
api = twitter.Api(CONSUMER_KEY,
          CONSUMER_SECRET,
          ACCESS_TOKEN,
          ACCESS_TOKEN_SECRET)

uri = "bolt://neo4j:7687"
sys.stderr.write("GraphDatabase.driver\n")
neo4j = GraphDatabase.driver(uri) # enable auth => , auth=("neo4j", "root"))
sys.stderr.write("session\n")
session = neo4j.session()

sys.stderr.write("tb\n")
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

# http://boundingbox.klokantech.com/
# A list of Longitude,Latitude pairs specifying bounding boxes for the tweets’ origin
FRANCE = ['-5.0006170181','42.1758579539', '8.6452145941','51.1825646283']
sys.stderr.write("Waiting 4 tweet\n")
nbTweets = 0
for tweet in api.GetStreamFilter(locations=FRANCE):
    #sys.stderr.write("NEW TWEET =============================================\n")
    #sys.stderr.write(tweet)
    try:
        sys.stderr.flush()
        lang = tweet['lang']
        if lang != 'fr':
            continue
        tweet_id = str(tweet['id'])
        user_id  = str(tweet['user']['id']) 
        text = tweet['text']
        blob = tb(text)
        polarity, subjectivity = blob.sentiment
        sys.stderr.write("Inserting new messsage {} from {}\n".format(tweet_id, user_id))
        request = """
        CREATE (t:TWEET {{ ID:{tweet_id}, TEXT:'{text}', POLARITY:{polarity}, SUBJECTIVITY:{subjectivity} }})
        MERGE (u:USER {{ID:{user_id}}}) SET u.NAME='{user_name}', u.PSEUDO='{pseudo}', u.LOCATION='{location}'
        CREATE (u)-[:WRITE]->(t)
        """.format(tweet_id = tweet_id, text = escape(text), polarity = polarity, subjectivity = subjectivity,
        user_id = user_id, user_name = escape(tweet['user']['name']), pseudo = escape(tweet['user']['screen_name']), location = escape(tweet['user']['location']))
        
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
                sys.stderr.write(user_id + " reply to " + str(original_user) + " about " + str(original_tweet))
                
        
        if tweet['entities'] and tweet['entities']['user_mentions'] and len(tweet['entities']['user_mentions']) > 0:
            for mention in tweet['entities']['user_mentions']:
                request += """
        MERGE (m{mention_to}:USER {{ID:{mention_to}}})
        CREATE (t)-[:MENTION]->(m{mention_to})
        """.format(mention_to=mention['id'])
        
        # sys.stderr.write("Executing = " + request)
        result = session.run(request)
        # sys.stderr.write(result)
        nbTweets += 1
        if nbTweets % 10 == 0:
            sys.stderr.write("Number of tweets : " + str(nbTweets))    
    except Exception as e:
        sys.stderr.write("Exception: {0}".format(e))
        sys.stderr.write(str(tweet))
        
