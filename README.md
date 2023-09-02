# Constrained adversarial attacks

Experimental setup for introducing constraints to universal adversarial attacks.

### Usage

Install dependencies

```
pip install -r requirements.txt
```

Run all experiments for all configuration options:

```
make all
```

For more detailed help, run

```
python3 -m exp --help
```

### Workflow

```
             ○      
             │      
      ┌──────┴─────┐ 
      │ parse args │
      └──────┬─────┘             
             │
   ┌─────────┴────────┐
   │ setup experiment │
   └─────────┬────────┘
             │
     preprocess input
  init classifier, attack
  create validation model
             │
    ┌────────┴───────┐             
    │ run experiment │
    │    k times:    │
    └────────┬───────┘
             │
        train model
       apply attack 
      evaluate/score
             │
    ┌────────┴───────┐             
    │ end experiment │
    └────────┬───────┘
             │
     ┌───────┴──────┐             
     │ write result │
     └───────┬──────┘
             │
             ◎
```


### Development instructions

First install all dev dependencies:

```
pip install -r requirements-dev.txt
```

Available code quality checks

```
make test    -- Run unit tests
make lint    -- Run linter
make dev     -- Test and lint, all at once
```