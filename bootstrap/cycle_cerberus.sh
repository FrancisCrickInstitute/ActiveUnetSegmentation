cerberus create -c boot-crb.json
cerberus attach -c boot-crb.json
cerberus inspect -c boot-crb.json
cerberus train -c boot-crb.json
cerberus predict boot-crb-latest.h5 sample.tif
