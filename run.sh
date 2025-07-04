dirdate=$(date +%Y-%m-%d-%H-%M-%S)

# ECP proxy search
nohup python evolve_proxy.py --ptype 'nasbench201' --GPU '1' --seed 1 --dataset cifar10 --batch_size 128 --num_train_samples 1000 --pop_size 20 --num_generations 40 >> ./proxy_search_nb201-c10-$dirdate.log 2>&1 &
#nohup python evolve_proxy.py --ptype 'nasbench201' --GPU '1' --seed 1 --dataset cifar100 --batch_size 128 --num_train_samples 1000 --pop_size 20 --num_generations 40 >> ./proxy_search_nb201-c100-$dirdate.log 2>&1 &
#nohup python evolve_proxy.py --ptype 'nasbench201' --GPU '2' --seed 1 --dataset ImageNet16-120 --batch_size 128 --num_train_samples 1000 --pop_size 20 --num_generations 40 >> ./proxy_search_nb201-in16-$dirdate.log 2>&1 &
#nohup python evolve_proxy.py --ptype 'nasbenchsss' --GPU '2' --seed 1 --dataset cifar10 --batch_size 128 --num_train_samples 1000 --pop_size 20 --num_generations 40 >> ./proxy_search_nbsss-c10-$dirdate.log 2>&1 &
#nohup python evolve_proxy.py --ptype 'nasbenchsss' --GPU '1' --seed 1 --dataset cifar100 --batch_size 128 --num_train_samples 1000 --pop_size 20 --num_generations 40 >> ./proxy_search_nbsss-c100-$dirdate.log 2>&1 &
#nohup python evolve_proxy.py --ptype 'nasbenchsss' --GPU '0' --seed 1 --dataset ImageNet16-120 --batch_size 128 --num_train_samples 1000 --pop_size 20 --num_generations 40 >> ./proxy_search_nbsss-in16-$dirdate.log 2>&1 &

