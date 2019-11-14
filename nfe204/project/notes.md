# Présentation du sujet : Les influencers sur internet

## 

## Code

pip install python-twitter
pip install neo4j-driver


# Neo4J

## TL;DR

Neo4j is a Database - use it to reliably store information and find it later
Neo4j’s data model is a Graph, in particular a Property Graph
Cypher is Neo4j’s graph query language (SQL for graphs!)

## Glossaire

 - Graph database
 - Bolt protocol
 - Cypher

## Concept 

A graph is composed of two elements: a node and a relationship.

Each node represents an entity (a person, place, thing, category or other piece of data), and each relationship represents how two nodes are associated

 - node : Graph data records. Similar nodes can have different properties
 - properties : value associated to a node. Properties are simple name/value pair. Properties can be strings, numbers, or booleans.
 - graph 
 - label : A node can have zero or more labels. Labels do not have any properties
 - relationship : Connect nodes
   * Relationships always have direction
   * Relationships always have a type
   * Relationships form patterns of data

## Bases de données graph

### Qu'est-ce qu'une base de données graph

Une base de données graphe est un système de gestion de base de données dont le modèle est le graphe.
Les relations entre entités sont la pierre angulaire du système.
C'est ça qui en fait la distinction par rapport aux autres systèmes qui soient SQL ou NoSQL.

### Autres exemples

### Pourquoi Neo4J?

### Qui utilise les bases de données graph? Neo4J?

## Ecosystème

### Neo4j

#### Historique
 - Pourquoi?
 - Origine du nom

#### Architecture


### Neo4j Desktop and Neo4j Enterprise Edition for Developers
https://neo4j.com/download/
https://neo4j.com/download/other-releases/#releases


## Installation du serveur

	$ mkdir -p ~/neo4j/data
	$ docker run \
	    --publish=7474:7474 --publish=7687:7687 \
	    --volume=$HOME/neo4j/data:/data \
	    neo4j

Contient un tutorial intégré

## Accès

 - via le navigateur : http://localhost:7474
 - via cypher-shell (sous ubuntu : sudo apt-get install neo4j-client)
 - Neo4J Desktop
 - via un développement spécifique (cf. Drivers)

## Requêtage

How this is related to that?
Special language to manipulate graph 
https://neo4j.com/docs/cypher-refcard/current/

### Insérer des données

#### Noeuds	

```
CREATE (TheMatrix:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})
```

#### Relations 

```
CREATE (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix)
```

### Trouver des données

#### Requêtes simples

```
MATCH (tom {name: "Tom Hanks"}) RETURN tom
MATCH (nineties:Movie) WHERE nineties.released >= 1990 AND nineties.released < 2000 RETURN nineties.title
```

#### Requêtes par relation

##### Liaisons simples

```
MATCH (tom:Person {name: "Tom Hanks"})-[:ACTED_IN]->(tomHanksMovies) RETURN tom,tomHanksMovies
MATCH (cloudAtlas {title: "Cloud Atlas"})<-[:DIRECTED]-(directors) RETURN directors.name
MATCH (tom:Person {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors) RETURN coActors.name
MATCH (people:Person)-[relatedTo]-(:Movie {title: "Cloud Atlas"}) RETURN people.name, Type(relatedTo), relatedTo
```

##### Liaisons complexes

MATCH (bacon:Person {name:"Kevin Bacon"})-[*1..4]-(hollywood) RETURN DISTINCT hollywood
MATCH p=shortestPath(
	  (bacon:Person {name:"Kevin Bacon"})-[*]-(meg:Person {name:"Meg Ryan"})
	)
	RETURN p
```
Extend Tom Hanks co-actors, to find co-co-actors who haven't work with Tom Hanks...
MATCH (tom:Person {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors),
      (coActors)-[:ACTED_IN]->(m2)<-[:ACTED_IN]-(cocoActors)
	WHERE NOT (tom)-[:ACTED_IN]->()<-[:ACTED_IN]-(cocoActors) AND tom <> cocoActors
RETURN cocoActors.name AS Recommended, count(*) AS Strength ORDER BY Strength DESC
```

Find someone to introduce Tom Hanks to Tom Cruise
```
MATCH (tom:Person {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors),
	      (coActors)-[:ACTED_IN]->(m2)<-[:ACTED_IN]-(cruise:Person {name:"Tom Cruise"})
	RETURN tom, m, coActors, m2, cruise
```

### TODO

**Aggregations?**
**INDEX?**

## En quoi Neo4j est un système NoSQL

Neo4J supports ACID
Transcation

### From tradionnal SQL/NoSQL to graph databases

### Requêtage des données interconnectées plus efficaces

### Schema-free
http://neo4j.com/docs/developer-manual/current/cypher/schema/constraints/#query-constraint-prop-exist-nodes


### Sharding/Partionnement
Scale Cue de Richardson
http://microservices.io/articles/scalecube.html

 - Cache sharding
 - Not supported?

### Clustering

Clustering : edition entreprise
https://dzone.com/articles/introducing-neo4j-31-now-in-beta-release
#### Causal Cluster

http://neo4j.com/docs/operations-manual/current/clustering/causal-clustering/introduction/
 - RAFT algorithm (https://raft.github.io/, visual => http://thesecretlivesofdata.com/raft/), idem **CONSUL**, + performant que PAXOS
 - If a majority of servers of the clusers (N/2 + 1) have accepted the transaction
 - Multiple writes, one slave

Example : https://neo4j.com/docs/operations-manual/current/installation/docker/#docker-cc

```
2017-12-12 16:31:45.448+0000 INFO  My connection info: [
	Discovery:   listen=0.0.0.0:5000, advertised=6f2409890a83:5000,
	Transaction: listen=0.0.0.0:6000, advertised=6f2409890a83:6000, 
	Raft:        listen=0.0.0.0:7000, advertised=6f2409890a83:7000, 
	Client Connector Addresses: bolt://localhost:7687,http://localhost:7474,https://localhost:7473
]
```

```
docker network create --driver=bridge cluster

docker run --name=core1 --detach --network=cluster \
 --env NEO4J_AUTH=neo4j/root \
 --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
 --publish=5000:5000 --publish=6000:6000 --publish=7000:7000 \
 --publish=7474:7474 --publish=7687:7687 \
 --env=NEO4J_dbms_mode=CORE \
 --env=NEO4J_causalClustering_expectedCoreClusterSize=3 \
 --env=NEO4J_causalClustering_initialDiscoveryMembers=core1:5000,core2:5000,core3:5000 \
 neo4j:3.3-enterprise

docker run --name=core2 --detach --network=cluster \
 --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
 --env=NEO4J_dbms_mode=CORE \
 --env=NEO4J_causalClustering_expectedCoreClusterSize=3 \
 --env=NEO4J_causalClustering_initialDiscoveryMembers=core1:5000,core2:5000,core3:5000 \
 neo4j:3.3-enterprise

docker run --name=core3 --detach --network=cluster \
 --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
 --env=NEO4J_dbms_mode=CORE \
 --env=NEO4J_causalClustering_expectedCoreClusterSize=3 \
 --env=NEO4J_causalClustering_initialDiscoveryMembers=core1:5000,core2:5000,core3:5000 \
 neo4j:3.3-enterprise
```

https://neo4j.com/docs/operations-manual/current/clustering/causal-clustering/setup-new-cluster/

http://localhost:7474/browser/

**:sysinfo** affiche les informations sur le cluster

Récupération information cluster : docker inspect --format '{{ .NetworkSettings.Networks.cluster.IPAddress }}' core2


#### High-availability cluster

Master/Slave
http://neo4j.com/docs/operations-manual/current/clustering/high-availability/architecture/

One master, multiple slaves

#### HA CC. vs
Neo4j High Availability refers to an approach for scaling the number of requests to which Neo4j can respond. Neo4j HA implements a master slave with replication clustering model for high availability scaling. This means that all writes go to the "master" server (or are proxied to master from the slaves) and the update is synchronized among the slave servers. Reads can be sent to any server in the cluster, scaling out the number of requests to which the database can respond.

Compare this to distributed computing, which is a general term to describe how computation operations can be done in parallel across a large number of machines. One key difference is the concept of data sharding. With Neo4j each server in the cluster contains a full copy of the graph, whereas with a distributed filesystem such as HDFS, the data is sharded and each machine stores only a small piece of the entire dataset.

Part of the reason Neo4j does not shard the graph is that since a graph is a highly connected data structure, traversing through a distributed/sharded graph would involve lots of network latency as the traversal "hops" from machine to machine.

### Reprise sur panne

### Volume

 - Neo4j can store billions of nodes

# Neo4j pour notre problème

## Modélisation

## Requêtes 

**docker-compose**?

# Annexes

## Bibliographie
 - Neo4J in action, Manning
 - Graph databases, O'Reilly
