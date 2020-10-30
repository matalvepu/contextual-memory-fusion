module load mongodb/3.4.10
port=`findport 27017`
mongod --dbpath /scratch/mhasan8/mongodb_storage/multi_humor --port $port > mongod.log &
