# CassieRun

## Abstract
Cassie Environment with MBPO method. 

## Running experiments
```bash
python main.py algorithm=mbpo overrides=mbpo_cassie
```

Result can be checked in 
```
exp/                                            # directory with all of the saved result
└── mbpo                                        # algorithm name
    └── default                         
        └── cassie                              # environment name
            └── save_name                       # unique save name by date
                └── save_name                   # unique save name 
                    ├── model.pth               # model for algo
                    ├── results.csv             # result
                    └── .hydra                  # readable hyperparameters
                        └── config.yaml         # readable hyperparameters for this run
```
