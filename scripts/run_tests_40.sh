#! /bin/bash
# Runs simulation tests for opt-trajectory project
#
# Source code should be located in opt-trajectory/src

DIR=$( dirname "${BASH_SOURCE[0]}" )

echo "${DIR}"

source "${DIR}/../../env_drone/bin/activate"

list_cond=( "--B 10 --vmax 10 --amax 30 --Ntests 10 --Nq 40" \
            "--B 10 --vmax 20 --amax 30 --Ntests 10 --Nq 40" \
            "--B 10 --vmax 30 --amax 30 --Ntests 10 --Nq 40" \
            "--B 10 --vmax 10 --amax 10 --Ntests 10 --Nq 40" \
            "--B 10 --vmax 20 --amax 10 --Ntests 10 --Nq 40" \
            "--B 10 --vmax 30 --amax 10 --Ntests 10 --Nq 40" )

for config in "${list_cond[@]}"
do
    echo "${config}"
    echo "python3 ${DIR}/../src/main_sim.py ${config}"
    python3 "${DIR}/../src/main_sim.py" ${config}
done
