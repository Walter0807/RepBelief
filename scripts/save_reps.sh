method="0shot"
max_tokens=20
num=200
offset=0
temperature=0
model="Mistral-7B-Instruct-v0.2"
# list of init beliefs
init_beliefs=$1
# list of conditions
conditions="false_belief true_belief"
# list of variables
variables=$2

for init_belief in $init_beliefs
do
    for condition in $conditions
    do
        for variable in $variables
        do
            python save_reps.py -n $num --init_belief $init_belief --method $method --condition $condition --variable $variable --mcq --model_name $model --max_tokens $max_tokens --temperature $temperature --offset $offset --verbose
        done
    done
done