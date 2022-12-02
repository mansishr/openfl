#!/bin/bash
cd /home/lab_phe3223/mansi/stg/openfl
for COMMIT in {1..100};
do
        echo ' ' >> tests/github/test_hello_federation.sh
        git commit -a -m "Test"
        git push origin test_failing_straggler_handling
        sleep 1m
done
