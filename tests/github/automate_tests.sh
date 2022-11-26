#!/bin/bash
cd /home/lab_phe3223/mansi/stg/openfl
for COMMIT in {1..3};
do
        echo ' ' >> tests/github/test_hello_federation.sh
        git commit -a -m "Test"
        git push mansishr test_failing_straggler_handling
done