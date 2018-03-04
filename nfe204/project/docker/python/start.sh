#!/bin/sh

pip3 install python-twitter
pip3 install textblob
pip3 install textblob_fr
pip3 install neo4j-driver

date
echo "Waiting 3 min"
sleep 60
echo "starting tweet-reader.py"
python --version
ls -lrt /script/tweet-reader.py
chmod +x /script/tweet-reader.py
/script/tweet-reader.py
